import os
import json
import asyncio
import logging
import threading
import tempfile
import re
from datetime import datetime
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import aiohttp
import requests
import joblib
import portalocker
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from huggingface_hub import HfApi

# ══════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════
#  SECRETS FROM ENVIRONMENT
# ══════════════════════════════════════════════
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
PHANTOM_KEY      = os.getenv("PHANTOM_KEY", "")
HF_TOKEN         = os.getenv("HF_TOKEN", "")

# ══════════════════════════════════════════════
#  FILE PATHS
# ══════════════════════════════════════════════
CONFIG_PATH       = "config.json"
ACTIVE_TRADE_PATH = "active_trade.json"
LSTM_MODEL_PATH   = "models/lstm_final.keras"
SCALER_PATH       = "models/scaler.pkl"
XGB_MODEL_PATH    = "models/xgboost_model.pkl"
SEQ_LEN           = 60

# ══════════════════════════════════════════════
#  TIMING
# ══════════════════════════════════════════════
SPIKE_INTERVAL_SEC = 300   # every 5 min
QUICK_INTERVAL_SEC = 900   # every 15 min
FULL_INTERVAL_SEC  = 3600  # every 60 min

# ══════════════════════════════════════════════
#  GLOBAL STATE
# ══════════════════════════════════════════════
last_prices: Dict[str, float] = {}

# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
DEFAULT_CONFIG = {
    "live_trading_enabled": False,
    "trading_pair": ["SOL/USDT", "BTC/USDT", "ETH/USDT"],
    "min_confidence": 65,
    "min_score": 6,
    "trailing_sl_pct": 0.02,
    "max_leverage": 3,
    "atr_chop_threshold": 1.5,
    "slippage_bps": 50,
    "max_priority_fee_lamports": 100000,
    "solana_rpc_url": "https://api.mainnet-beta.solana.com",
    "sentiment_block_buy": -0.4,
    "sentiment_block_sell": 0.4,
    "partial_take_profit_pct": 0.50,
    "spike_alert_pct": 3.0,
    "whale_volume_multiplier": 1.5,
    "signal_cycle_minutes": 60,
    "historical_baseline_profit": 1500,
    "hf_dataset_repo": "sol-matrix-bot/cryptoai-state-data",
    "performance_metrics": {
        "total_signals_generated": 0,
        "successful_signals": 0,
        "failed_signals": 0,
        "current_win_rate_pct": 0.0
    }
}

def load_config() -> Dict:
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        # Ensure performance_metrics always exists
        if "performance_metrics" not in cfg:
            cfg["performance_metrics"] = \
                DEFAULT_CONFIG["performance_metrics"].copy()
        return cfg
    except Exception as e:
        log.error(f"Config load error: {e} — using defaults")
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict):
    try:
        with portalocker.Lock(CONFIG_PATH, timeout=5):
            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=2)
    except Exception as e:
        log.error(f"Config save error: {e}")

def load_active_trade() -> Dict:
    try:
        with portalocker.Lock(ACTIVE_TRADE_PATH, timeout=5):
            with open(ACTIVE_TRADE_PATH, "r") as f:
                return json.load(f)
    except:
        return {}

def save_active_trade(data: Dict):
    try:
        with portalocker.Lock(ACTIVE_TRADE_PATH, timeout=5):
            with open(ACTIVE_TRADE_PATH, "w") as f:
                json.dump(data, f, indent=2)
    except Exception as e:
        log.error(f"Trade save error: {e}")

# ══════════════════════════════════════════════
#  HUGGING FACE STATE SYNC
# ══════════════════════════════════════════════
def save_state_to_hf(config: Dict):
    if not HF_TOKEN:
        return
    try:
        api     = HfApi(token=HF_TOKEN)
        payload = json.dumps(
            config.get("performance_metrics", {}),
            indent=2
        )
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            f.write(payload)
            tmp = f.name
        api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo="performance_metrics.json",
            repo_id=config.get(
                "hf_dataset_repo",
                "sol-matrix-bot/cryptoai-state-data"
            ),
            repo_type="dataset"
        )
        os.unlink(tmp)
        log.info("✅ Metrics synced to Hugging Face")
    except Exception as e:
        log.warning(f"HF sync error: {e}")

# ══════════════════════════════════════════════
#  TELEGRAM
# ══════════════════════════════════════════════
def _sync_send_message(
    token: str, chat_id: str, text: str
) -> Tuple[int, dict]:
    url  = f"https://api.telegram.org/bot{token.strip()}/sendMessage"
    resp = requests.post(
        url,
        json={
            "chat_id":    chat_id.strip(),
            "text":       text,
            "parse_mode": "HTML"
        },
        timeout=15
    )
    return resp.status_code, resp.json()

def _sync_send_photo(
    token: str, chat_id: str,
    path: str, caption: str
) -> Tuple[int, dict]:
    url = f"https://api.telegram.org/bot{token.strip()}/sendPhoto"
    with open(path, 'rb') as photo:
        resp = requests.post(
            url,
            data={
                "chat_id":    chat_id.strip(),
                "caption":    caption,
                "parse_mode": "HTML"
            },
            files={"photo": photo},
            timeout=25
        )
    return resp.status_code, resp.json()

async def send_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("⚠️ Telegram secrets missing!")
        return
    for attempt in range(3):
        try:
            status, data = await asyncio.to_thread(
                _sync_send_message,
                TELEGRAM_TOKEN,
                TELEGRAM_CHAT_ID,
                text
            )
            if status == 200:
                log.info("✅ Telegram message sent!")
                return
            log.warning(
                f"⚠️ Telegram {status}: "
                f"{data.get('description','unknown')}"
            )
        except Exception as e:
            log.warning(f"⚠️ Telegram attempt {attempt+1}: {e}")
            await asyncio.sleep(5)
    log.error("❌ Telegram message failed all attempts")

async def send_photo(path: str, caption: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    if not os.path.exists(path):
        log.warning("⚠️ Chart missing — sending text")
        await send_message(caption)
        return
    for attempt in range(3):
        try:
            status, data = await asyncio.to_thread(
                _sync_send_photo,
                TELEGRAM_TOKEN,
                TELEGRAM_CHAT_ID,
                path,
                caption
            )
            if status == 200:
                log.info("✅ Chart sent to Telegram!")
                return
            log.warning(
                f"⚠️ Photo {status}: "
                f"{data.get('description','unknown')}"
            )
        except Exception as e:
            log.warning(f"⚠️ Photo attempt {attempt+1}: {e}")
            await asyncio.sleep(5)
    log.error("❌ Photo failed — sending text only")
    await send_message(caption)

# ══════════════════════════════════════════════
#  MARKET DATA
# ══════════════════════════════════════════════
COINGECKO_IDS = {
    "SOLUSDT": "solana",
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum"
}
KRAKEN_PAIRS = {
    "SOLUSDT": "SOLUSD",
    "BTCUSDT": "XBTUSD",
    "ETHUSDT": "ETHUSD"
}

async def fetch_current_price(symbol: str) -> Optional[float]:
    clean   = symbol.replace("/", "").upper()
    coin_id = COINGECKO_IDS.get(clean)
    if not coin_id:
        return None
    try:
        url = (
            "https://api.coingecko.com/api/v3/simple/price"
            f"?ids={coin_id}&vs_currencies=usd"
        )
        async with aiohttp.ClientSession() as s:
            async with s.get(
                url, timeout=aiohttp.ClientTimeout(total=8)
            ) as r:
                if r.status == 200:
                    d = await r.json()
                    return float(d[coin_id]["usd"])
    except Exception as e:
        log.warning(f"⚠️ Price error {symbol}: {e}")
    return None

async def _coingecko_candles(
    clean: str
) -> Optional[pd.DataFrame]:
    coin_id = COINGECKO_IDS.get(clean)
    if not coin_id:
        return None
    url = (
        f"https://api.coingecko.com/api/v3/coins/"
        f"{coin_id}/ohlc?vs_currency=usd&days=14"
    )
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                url, timeout=aiohttp.ClientTimeout(total=12)
            ) as r:
                if r.status == 200:
                    raw = await r.json()
                    df  = pd.DataFrame(
                        raw,
                        columns=[
                            "timestamp","open",
                            "high","low","close"
                        ]
                    )
                    df = df.astype(float)
                    df["volume"]    = 1.0
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"].astype(int),
                        unit="ms"
                    )
                    return df.sort_values(
                        "timestamp"
                    ).reset_index(drop=True)
                log.warning(
                    f"⚠️ CoinGecko {r.status} for {clean}"
                )
    except Exception as e:
        log.warning(f"⚠️ CoinGecko error {clean}: {e}")
    return None

async def _kraken_candles(
    clean: str
) -> Optional[pd.DataFrame]:
    pair = KRAKEN_PAIRS.get(clean)
    if not pair:
        return None
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                "https://api.kraken.com/0/public/OHLC",
                params={"pair": pair, "interval": 60},
                timeout=aiohttp.ClientTimeout(total=12)
            ) as r:
                d = await r.json()
                if d.get("error"):
                    raise ValueError(str(d["error"]))
                key  = list(d["result"].keys())[0]
                rows = d["result"][key]
                df   = pd.DataFrame(rows, columns=[
                    "timestamp","open","high","low",
                    "close","vwap","volume","count"
                ])
                df = df[
                    ["timestamp","open","high",
                     "low","close","volume"]
                ].astype({
                    "open": float, "high": float,
                    "low": float,  "close": float,
                    "volume": float
                })
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"].astype(int), unit="s"
                )
                return df.sort_values(
                    "timestamp"
                ).reset_index(drop=True)
    except Exception as e:
        log.warning(f"⚠️ Kraken error {clean}: {e}")
    return None

async def fetch_candles(
    symbol: str
) -> Optional[pd.DataFrame]:
    clean = symbol.replace("/", "").upper()

    df = await _coingecko_candles(clean)
    if df is not None and len(df) >= SEQ_LEN + 20:
        log.info(f"✅ CoinGecko → {symbol} ({len(df)} rows)")
        return df

    log.warning(f"⚠️ CoinGecko failed → Kraken for {symbol}")
    df = await _kraken_candles(clean)
    if df is not None and len(df) >= SEQ_LEN + 20:
        log.info(f"✅ Kraken → {symbol} ({len(df)} rows)")
        return df

    log.error(f"🚨 All sources failed for {symbol}")
    return None

# ══════════════════════════════════════════════
#  TECHNICAL INDICATORS
# ══════════════════════════════════════════════
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"]         = RSIIndicator(
                            df["close"], window=14
                        ).rsi()
    bb                = BollingerBands(df["close"], window=20)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_width"]    = bb.bollinger_wband()
    df["atr"]         = AverageTrueRange(
                            df["high"], df["low"],
                            df["close"], window=14
                        ).average_true_range()
    macd              = MACD(df["close"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["ema_20"]      = EMAIndicator(
                            df["close"], window=20
                        ).ema_indicator()
    df["ema_50"]      = EMAIndicator(
                            df["close"], window=50
                        ).ema_indicator()
    return df.dropna().reset_index(drop=True)

# ══════════════════════════════════════════════
#  PILLAR 4 — MARKET REGIME FILTER
# ══════════════════════════════════════════════
def is_trending(df: pd.DataFrame, config: Dict) -> bool:
    row       = df.iloc[-1]
    threshold = config.get("atr_chop_threshold", 1.5)
    trending  = (
        row["bb_width"] >= 0.05
        and row["atr"] >= threshold
    )
    log.info(
        f"📊 Regime: {'TRENDING ✅' if trending else 'CHOP 🔄'} "
        f"| BB Width: {row['bb_width']:.4f} "
        f"| ATR: {row['atr']:.4f}"
    )
    return trending

# ══════════════════════════════════════════════
#  PILLAR 2 — WHALE / LIQUIDITY SWEEP
# ══════════════════════════════════════════════
def detect_whale_sweep(df: pd.DataFrame) -> bool:
    recent  = df.tail(5)
    vol_avg = df["volume"].rolling(20).mean().iloc[-1]
    if pd.isna(vol_avg) or vol_avg <= 0:
        vol_avg = 1.0
    for _, c in recent.iterrows():
        body = abs(c["close"] - c["open"])
        low_wick = (
            min(c["open"], c["close"]) - c["low"]
        )
        bullish_candle = c["close"] > c["open"]
        big_wick       = low_wick > body * 2
        volume_spike   = c["volume"] >= vol_avg * 1.1
        if bullish_candle and big_wick and volume_spike:
            log.info("🐳 Whale liquidity sweep detected!")
            return True
    return False

# ══════════════════════════════════════════════
#  PILLAR 3 — MACRO BIAS
# ══════════════════════════════════════════════
MACRO_BIAS = "neutral"  # "bullish" | "bearish" | "neutral"

def macro_allows(direction: str) -> bool:
    if MACRO_BIAS == "bearish" and direction == "LONG":
        log.warning("🚫 Macro bias bearish — blocking LONG")
        return False
    if MACRO_BIAS == "bullish" and direction == "SHORT":
        log.warning("🚫 Macro bias bullish — blocking SHORT")
        return False
    return True

# ══════════════════════════════════════════════
#  PILLAR 5 — NEWS SENTIMENT (VADER)
# ══════════════════════════════════════════════
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://coindesk.com/arc/outboundfeeds/rss/"
]

async def fetch_sentiment() -> float:
    vader     = SentimentIntensityAnalyzer()
    headlines = []
    async with aiohttp.ClientSession() as s:
        for url in RSS_FEEDS:
            try:
                async with s.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=8)
                ) as r:
                    html  = await r.text()
                    found = re.findall(
                        r'<title>(.*?)</title>', html
                    )[2:12]
                    headlines.extend(found)
            except Exception as e:
                log.warning(f"⚠️ Feed error {url}: {e}")

    if not headlines:
        log.warning("⚠️ No headlines — sentiment = 0.0")
        return 0.0

    scores = [
        vader.polarity_scores(h)["compound"]
        for h in headlines[:15]
    ]
    avg = round(sum(scores) / len(scores), 4)
    log.info(
        f"📰 Sentiment: {avg} ({len(scores)} headlines)"
    )
    return avg

def sentiment_allows(
    score: float, direction: str, config: Dict
) -> bool:
    block_buy  = config.get("sentiment_block_buy", -0.4)
    block_sell = config.get("sentiment_block_sell",  0.4)
    if score < block_buy and direction == "LONG":
        log.warning(
            f"🚫 Sentiment {score} blocking LONG (buy)"
        )
        return False
    if score > block_sell and direction == "SHORT":
        log.warning(
            f"🚫 Sentiment {score} blocking SHORT (sell)"
        )
        return False
    return True

# ══════════════════════════════════════════════
#  FEAR & GREED INDEX
# ══════════════════════════════════════════════
async def fetch_fear_greed() -> Dict:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                "https://api.alternative.me/fng/",
                timeout=aiohttp.ClientTimeout(total=8)
            ) as r:
                d   = await r.json()
                val = int(d["data"][0]["value"])
                cls = d["data"][0]["value_classification"]
                log.info(f"😨 Fear & Greed: {val} ({cls})")
                return {"value": val, "label": cls}
    except Exception as e:
        log.warning(f"⚠️ Fear & Greed error: {e}")
    return {"value": 50, "label": "Neutral"}

# ══════════════════════════════════════════════
#  PILLAR 1 — AI BRAIN (LSTM + XGBOOST)
# ══════════════════════════════════════════════
def _build_lstm(shape: Tuple) -> tf.keras.Model:
    m = Sequential([
        LSTM(128, return_sequences=True, input_shape=shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1,  activation="sigmoid")
    ])
    m.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return m

LSTM_FEATURES = [
    "close","volume","rsi","macd","bb_width","atr"
]
XGB_FEATURES = [
    "rsi","macd","bb_width",
    "atr","ema_20","ema_50","volume"
]

def train_models(df: pd.DataFrame):
    log.info("🧠 Training LSTM + XGBoost models...")
    os.makedirs("models", exist_ok=True)

    # ── LSTM ──────────────────────────────────
    data   = df[LSTM_FEATURES].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y   = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i])
        y.append(
            1 if df["close"].iloc[i] >
            df["close"].iloc[i - 1] else 0
        )
    X, y  = np.array(X), np.array(y)
    model = _build_lstm((X.shape[1], X.shape[2]))
    model.fit(
        X, y, epochs=5,
        batch_size=32, verbose=0
    )
    model.save(LSTM_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    log.info("✅ LSTM trained and saved")

    # ── XGBoost ───────────────────────────────
    X_x  = df[XGB_FEATURES].values[:-1]
    y_x  = (
        df["close"].shift(-1) > df["close"]
    ).astype(int).values[:-1]
    xgbm = xgb.XGBClassifier(
        n_estimators=100, max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        use_label_encoder=False
    )
    xgbm.fit(X_x, y_x)
    joblib.dump(xgbm, XGB_MODEL_PATH)
    log.info("✅ XGBoost trained and saved")

def ai_decision(
    df: pd.DataFrame
) -> Tuple[Optional[str], float]:
    # Auto-train if models missing
    models_exist = (
        os.path.exists(LSTM_MODEL_PATH)
        and os.path.exists(XGB_MODEL_PATH)
        and os.path.exists(SCALER_PATH)
    )
    if not models_exist:
        log.warning("⚠️ Models missing — training now...")
        train_models(df)

    try:
        lstm   = load_model(LSTM_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        xgbm   = joblib.load(XGB_MODEL_PATH)

        # LSTM prediction
        raw    = df[LSTM_FEATURES].values[-SEQ_LEN:]
        scaled = scaler.transform(raw)
        p_lstm = float(
            lstm.predict(
                np.array([scaled]), verbose=0
            )[0][0]
        )

        # XGBoost prediction
        p_xgb = float(
            xgbm.predict_proba(
                df[XGB_FEATURES].values[-1:]
            )[0][1]
        )

        dir_lstm = "LONG" if p_lstm > 0.5 else "SHORT"
        dir_xgb  = "LONG" if p_xgb  > 0.5 else "SHORT"

        log.info(
            f"🧠 LSTM={dir_lstm}({p_lstm:.2%}) | "
            f"XGB={dir_xgb}({p_xgb:.2%})"
        )

        # Both must agree — Pillar 1 rule
        if dir_lstm != dir_xgb:
            log.warning("⚠️ Models disagree — no signal!")
            return None, 0.0

        # Average confidence
        if dir_lstm == "LONG":
            conf = (p_lstm + p_xgb) / 2
        else:
            conf = ((1 - p_lstm) + (1 - p_xgb)) / 2

        return dir_lstm, round(conf, 4)

    except Exception as e:
        log.error(f"🚨 AI decision error: {e}")
        return None, 0.0

# ══════════════════════════════════════════════
#  SIGNAL SCORING SYSTEM
# ══════════════════════════════════════════════
def score_signal(
    direction:  str,
    confidence: float,
    sentiment:  float,
    whale:      bool,
    fg:         Dict,
    df:         pd.DataFrame
) -> int:
    score = 0
    row   = df.iloc[-1]

    # AI confidence — max 3 pts
    if confidence >= 0.75:   score += 3
    elif confidence >= 0.65: score += 2
    else:                    score += 1

    # Whale sweep — max 2 pts
    if whale: score += 2

    # Sentiment — max 1 pt
    if direction == "LONG"  and sentiment >  0.1: score += 1
    if direction == "SHORT" and sentiment < -0.1: score += 1

    # RSI oversold/overbought — max 1 pt
    rsi = row.get("rsi", 50)
    if direction == "LONG"  and rsi < 35: score += 1
    if direction == "SHORT" and rsi > 65: score += 1

    # Fear & Greed — max 1 pt
    fgv = fg.get("value", 50)
    if direction == "LONG"  and fgv < 30: score += 1
    if direction == "SHORT" and fgv > 70: score += 1

    # EMA trend alignment — max 1 pt
    e20 = row.get("ema_20", 0)
    e50 = row.get("ema_50", 0)
    if direction == "LONG"  and e20 > e50: score += 1
    if direction == "SHORT" and e20 < e50: score += 1

    final = min(score, 10)
    log.info(
        f"📊 Score: {final}/10 | "
        f"Conf: {confidence:.1%} | "
        f"RSI: {rsi:.1f} | "
        f"Whale: {whale} | "
        f"FG: {fgv}"
    )
    return final

# ══════════════════════════════════════════════
#  PILLAR 6 & 7 — TARGETS + TRAILING STOP
# ══════════════════════════════════════════════
def calculate_targets(
    df: pd.DataFrame, direction: str
) -> Tuple[float, float, float]:
    entry = df.iloc[-1]["close"]
    atr   = df.iloc[-1]["atr"]
    if direction == "LONG":
        target    = entry + atr * 3
        stop_loss = entry - atr * 2
    else:
        target    = entry - atr * 3
        stop_loss = entry + atr * 2
    return round(entry, 6), round(target, 6), round(stop_loss, 6)

# ══════════════════════════════════════════════
#  CHART ENGINE
# ══════════════════════════════════════════════
BG = "#0d1117"
FG = "#c9d1d9"

def generate_chart(
    df:        pd.DataFrame,
    symbol:    str,
    direction: str,
    entry:     float,
    target:    float,
    stop_loss: float
) -> str:
    chart = df.tail(100).copy().set_index("timestamp")
    chart.index = pd.DatetimeIndex(chart.index)
    path  = f"/tmp/{symbol.replace('/','')}_chart.png"

    try:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(13, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            facecolor=BG
        )

        # ── Price panel ──
        ax1.plot(
            chart.index, chart["close"],
            color="#58a6ff", linewidth=1.5,
            label="Price", zorder=3
        )
        ax1.axhline(
            entry, color="#f0e68c",
            linestyle="--", linewidth=1.8,
            label=f"Entry  ${entry:,.4f}"
        )
        ax1.axhline(
            target, color="#3fb950",
            linestyle="--", linewidth=1.8,
            label=f"Target ${target:,.4f}"
        )
        ax1.axhline(
            stop_loss, color="#f85149",
            linestyle="--", linewidth=1.8,
            label=f"Stop   ${stop_loss:,.4f}"
        )
        fill_color = "#3fb950" if direction == "LONG" else "#f85149"
        ax1.fill_between(
            chart.index,
            stop_loss, target,
            color=fill_color, alpha=0.06
        )
        ax1.set_facecolor(BG)
        ax1.tick_params(colors=FG)
        ax1.yaxis.label.set_color(FG)
        for spine in ax1.spines.values():
            spine.set_edgecolor("#30363d")
        ax1.set_title(
            f"{'🚀' if direction == 'LONG' else '📉'} "
            f"{symbol}  |  {direction} SIGNAL  |  "
            f"{datetime.now().strftime('%H:%M  %d %b %Y')}",
            color=FG, fontsize=12, pad=10
        )
        ax1.legend(
            loc="upper left",
            facecolor="#161b22",
            labelcolor=FG,
            fontsize=9
        )
        ax1.grid(True, color="#21262d", linewidth=0.7)

        # ── RSI panel ──
        rsi_vals = chart["rsi"] if "rsi" in chart.columns \
                   else pd.Series([50] * len(chart))
        ax2.plot(
            chart.index, rsi_vals,
            color="#e3b341", linewidth=1.3
        )
        ax2.axhline(70, color="#f85149", linestyle="--", alpha=0.6)
        ax2.axhline(50, color="#8b949e", linestyle=":",  alpha=0.4)
        ax2.axhline(30, color="#3fb950", linestyle="--", alpha=0.6)
        ax2.fill_between(
            chart.index, rsi_vals, 70,
            where=(rsi_vals >= 70),
            color="#f85149", alpha=0.2
        )
        ax2.fill_between(
            chart.index, rsi_vals, 30,
            where=(rsi_vals <= 30),
            color="#3fb950", alpha=0.2
        )
        ax2.set_facecolor(BG)
        ax2.tick_params(colors=FG)
        ax2.set_ylabel("RSI", color=FG, fontsize=9)
        ax2.set_ylim(0, 100)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#30363d")
        ax2.grid(True, color="#21262d", linewidth=0.7)

        plt.tight_layout(h_pad=0.5)
        plt.savefig(
            path, dpi=110,
            bbox_inches="tight",
            facecolor=BG
        )
        plt.close(fig)
        log.info(f"✅ Chart saved → {path}")

    except Exception as e:
        log.error(f"❌ Chart error: {e}")
        plt.close("all")

    return path

# ══════════════════════════════════════════════
#  PILLAR 9 — JUPITER DEX TRADING
# ══════════════════════════════════════════════
SOL_MINT  = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

async def execute_trade(
    symbol:      str,
    direction:   str,
    amount_usdc: float,
    config:      Dict
) -> Optional[dict]:
    if not config.get("live_trading_enabled", False):
        log.info("📊 SIGNAL ONLY mode — no live trade")
        return None

    in_mint  = USDC_MINT if direction == "LONG" else SOL_MINT
    out_mint = SOL_MINT  if direction == "LONG" else USDC_MINT
    amount   = int(amount_usdc * 1_000_000)

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    "https://quote-api.jup.ag/v6/quote",
                    params={
                        "inputMint":   in_mint,
                        "outputMint":  out_mint,
                        "amount":      amount,
                        "slippageBps": config.get(
                            "slippage_bps", 50
                        )
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as r:
                    quote = await r.json()
                    log.info(
                        f"✅ Jupiter quote → {symbol}"
                    )
                    return quote
        except Exception as e:
            log.warning(
                f"⚠️ Jupiter attempt {attempt+1}: {e}"
            )
            await asyncio.sleep(5)
    log.error("🚨 Jupiter quote failed")
    return None

# ══════════════════════════════════════════════
#  LAYER 1 — SPIKE ALERT (every 5 min)
# ══════════════════════════════════════════════
async def spike_check(config: Dict):
    pairs     = config.get(
        "trading_pair",
        ["SOL/USDT","BTC/USDT","ETH/USDT"]
    )
    threshold = config.get("spike_alert_pct", 3.0)

    for pair in pairs:
        try:
            price = await fetch_current_price(pair)
            if price is None:
                continue

            key = pair.replace("/","").upper()

            if key in last_prices:
                prev   = last_prices[key]
                change = (price - prev) / prev * 100
                log.info(
                    f"💰 {pair}: ${price:.4f} "
                    f"({change:+.2f}%)"
                )

                if abs(change) >= threshold:
                    icon = "🚨" if abs(change) >= 5 else "⚡"
                    arrow = "📈" if change > 0 else "📉"
                    await send_message(
                        f"{icon} <b>SPIKE ALERT — {pair}</b>\n\n"
                        f"{arrow} Change:  "
                        f"<b>{change:+.2f}%</b>\n"
                        f"💰 Now:    "
                        f"<code>${price:,.4f}</code>\n"
                        f"💰 Before: "
                        f"<code>${prev:,.4f}</code>\n\n"
                        f"⏰ "
                        f"{datetime.now().strftime('%H:%M %d/%m/%Y')}"
                    )
            else:
                log.info(
                    f"💰 {pair}: ${price:.4f} (first read)"
                )

            last_prices[key] = price

        except Exception as e:
            log.warning(f"⚠️ Spike check {pair}: {e}")

# ══════════════════════════════════════════════
#  LAYER 2 — WHALE CHECK (every 15 min)
# ══════════════════════════════════════════════
async def whale_check(config: Dict):
    pairs = config.get(
        "trading_pair",
        ["SOL/USDT","BTC/USDT","ETH/USDT"]
    )
    log.info(
        f"🐳 Whale check — "
        f"{datetime.now().strftime('%H:%M')}"
    )
    for pair in pairs:
        try:
            df = await fetch_candles(pair)
            if df is None or len(df) < 25:
                continue
            df    = add_indicators(df)
            whale = detect_whale_sweep(df)
            if whale:
                row = df.iloc[-1]
                await send_message(
                    f"🐳 <b>WHALE SWEEP — {pair}</b>\n\n"
                    f"💰 Price: "
                    f"<code>${row['close']:,.4f}</code>\n"
                    f"📊 RSI:   <b>{row['rsi']:.1f}</b>\n"
                    f"📉 ATR:   <b>{row['atr']:.4f}</b>\n\n"
                    f"👀 Big wick + volume spike!\n"
                    f"Watch this closely!\n\n"
                    f"⏰ "
                    f"{datetime.now().strftime('%H:%M %d/%m/%Y')}"
                )
        except Exception as e:
            log.warning(f"⚠️ Whale check {pair}: {e}")

# ══════════════════════════════════════════════
#  LAYER 3 — FULL AI SIGNAL CYCLE (every 60 min)
# ══════════════════════════════════════════════
async def full_signal_cycle(config: Dict) -> int:
    pairs     = config.get(
        "trading_pair",
        ["SOL/USDT","BTC/USDT","ETH/USDT"]
    )
    min_score = config.get("min_score", 6)
    min_conf  = config.get("min_confidence", 65) / 100
    fired     = 0

    log.info("=" * 55)
    log.info(
        f"🔬 FULL AI CYCLE — "
        f"{datetime.now().strftime('%H:%M %d/%m/%Y')}"
    )
    log.info(
        f"   Mode: "
        f"{'🔴 LIVE' if config.get('live_trading_enabled') else '🟡 SIGNAL ONLY'}"
    )
    log.info("=" * 55)

    # Fetch shared data once
    sentiment, fg = await asyncio.gather(
        fetch_sentiment(),
        fetch_fear_greed()
    )

    for pair in pairs:
        log.info(f"\n💎 Analysing {pair}...")
        log.info("-" * 40)

        try:
            # Get candles
            df = await fetch_candles(pair)
            if df is None or len(df) < SEQ_LEN + 20:
                log.warning(f"⚠️ {pair}: Not enough data")
                continue

            # Add indicators
            df = add_indicators(df)

            # Pillar 4 — regime filter
            if not is_trending(df, config):
                log.info(f"🔄 {pair}: Chop — skipping")
                continue

            # Pillar 1 — AI brain
            direction, confidence = ai_decision(df)
            if direction is None:
                continue

            if confidence < min_conf:
                log.info(
                    f"❌ {pair}: Confidence "
                    f"{confidence:.1%} < {min_conf:.1%}"
                )
                continue

            # Pillar 3 — macro bias
            if not macro_allows(direction):
                continue

            # Pillar 5 — sentiment filter
            if not sentiment_allows(
                sentiment, direction, config
            ):
                continue

            # Pillar 2 — whale sweep
            whale = detect_whale_sweep(df)

            # Score
            score = score_signal(
                direction, confidence,
                sentiment, whale, fg, df
            )

            if score < min_score:
                log.info(
                    f"❌ {pair}: Score {score}/10 "
                    f"< {min_score}/10 minimum"
                )
                continue

            # Targets
            entry, target, stop = calculate_targets(
                df, direction
            )
            log.info(
                f"🚀 SIGNAL FIRED! {direction}\n"
                f"   Entry:  ${entry:,.4f}\n"
                f"   Target: ${target:,.4f}\n"
                f"   Stop:   ${stop:,.4f}"
            )

            # Chart
            chart = generate_chart(
                df, pair, direction,
                entry, target, stop
            )

            # Telegram alert
            emoji = "🚀" if direction == "LONG" else "📉"
            msg   = (
                f"{emoji} <b>{direction} — {pair}</b>\n\n"
                f"📍 Entry:      "
                f"<code>${entry:,.4f}</code>\n"
                f"🎯 Target:     "
                f"<code>${target:,.4f}</code>\n"
                f"🛑 Stop Loss:  "
                f"<code>${stop:,.4f}</code>\n\n"
                f"📊 Score:      <b>{score}/10</b>\n"
                f"🎯 Confidence: <b>{confidence:.1%}</b>\n"
                f"😨 Fear/Greed: "
                f"<b>{fg['value']} ({fg['label']})</b>\n"
                f"📰 Sentiment:  <b>{sentiment:+.3f}</b>\n"
                f"🐳 Whale:      "
                f"<b>{'YES ✅' if whale else 'NO ❌'}</b>\n\n"
                f"⏰ "
                f"{datetime.now().strftime('%H:%M %d/%m/%Y')}"
            )
            await send_photo(chart, msg)

            # Save trade state
            save_active_trade({
                "symbol":     pair,
                "direction":  direction,
                "entry":      entry,
                "target":     target,
                "stop_loss":  stop,
                "score":      score,
                "confidence": confidence,
                "sentiment":  sentiment,
                "timestamp":  datetime.now().isoformat(),
                "status":     "ACTIVE"
            })

            # Live trade execution
            if config.get("live_trading_enabled", False):
                await execute_trade(
                    pair, direction, 100, config
                )

            # Update metrics
            pm = config.setdefault(
                "performance_metrics",
                DEFAULT_CONFIG["performance_metrics"].copy()
            )
            pm["total_signals_generated"] = \
                pm.get("total_signals_generated", 0) + 1
            save_config(config)
            save_state_to_hf(config)
            fired += 1

        except Exception as e:
            log.error(f"🚨 Cycle error {pair}: {e}")

    log.info("=" * 55)
    log.info(f"✅ Cycle done — {fired} signal(s) fired")
    log.info("=" * 55)
    return fired

# ══════════════════════════════════════════════
#  MAIN 24/7 LOOP
# ══════════════════════════════════════════════
async def main():
    config = load_config()
    pairs  = config.get(
        "trading_pair",
        ["SOL/USDT","BTC/USDT","ETH/USDT"]
    )

    log.info("🚀 CryptoAI Bot Starting...")

    await send_message(
        "🤖 <b>CryptoAI Bot Online!</b>\n\n"
        f"📊 Tracking: {', '.join(pairs)}\n\n"
        "⚡ Spike check:    every <b>5 min</b>\n"
        "🐳 Whale check:    every <b>15 min</b>\n"
        "🧠 Full AI cycle:  every <b>60 min</b>\n\n"
        f"Mode: "
        f"{'🔴 LIVE TRADING' if config.get('live_trading_enabled') else '🟡 SIGNAL ONLY'}\n"
        f"⏰ "
        f"{datetime.now().strftime('%H:%M %d/%m/%Y')}"
    )

    tick = 0  # counts 5-min intervals

    while True:
        try:
            config = load_config()
            tick  += 1

            # Every 5 min — spike check
            await spike_check(config)

            # Every 15 min — whale check (tick 3,6,9,12...)
            if tick % 3 == 0:
                await whale_check(config)

            # Every 60 min — full AI cycle (tick 12,24,36...)
            if tick % 12 == 0:
                fired = await full_signal_cycle(config)
                pm    = config.get(
                    "performance_metrics", {}
                )
                await send_message(
                    f"✅ <b>Full Cycle Done</b>\n\n"
                    f"Signals fired: <b>{fired}</b>\n"
                    f"Total signals: "
                    f"<b>{pm.get('total_signals_generated',0)}</b>\n"
                    f"Win rate: "
                    f"<b>{pm.get('current_win_rate_pct',0):.1f}%</b>\n"
                    f"Next cycle in: <b>60 min</b>\n\n"
                    f"⏰ "
                    f"{datetime.now().strftime('%H:%M %d/%m/%Y')}"
                )

        except Exception as e:
            log.error(f"🚨 Main loop crash: {e}")
            await send_message(
                f"⚠️ <b>Bot Error</b>\n"
                f"<code>{str(e)[:200]}</code>\n"
                f"Auto-recovering in 60s..."
            )
            await asyncio.sleep(60)
            continue

        log.info(
            f"⏳ Tick #{tick} done — "
            f"sleeping 5 min..."
        )
        await asyncio.sleep(SPIKE_INTERVAL_SEC)

# ══════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("🛑 Bot stopped by user.")
