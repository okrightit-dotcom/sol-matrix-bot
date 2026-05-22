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

# ── Logging ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ── Secrets ────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PHANTOM_KEY      = os.getenv("PHANTOM_KEY")
HF_TOKEN         = os.getenv("HF_TOKEN")

# ── Paths ──────────────────────────────────────
CONFIG_PATH       = "config.json"
ACTIVE_TRADE_PATH = "active_trade.json"
LSTM_MODEL_PATH   = "models/lstm_final.keras"
SCALER_PATH       = "models/scaler.pkl"
XGB_MODEL_PATH    = "models/xgboost_model.pkl"
SEQ_LEN           = 60

# ── Timing Settings ────────────────────────────
SPIKE_CHECK_SECONDS  = 300   # 5 minutes
QUICK_CYCLE_SECONDS  = 900   # 15 minutes
FULL_CYCLE_SECONDS   = 3600  # 60 minutes

# ── Spike Tracker ──────────────────────────────
last_prices: Dict[str, float] = {}

# ── File Lock ──────────────────────────────────
file_lock = threading.Lock()

# ── Config ─────────────────────────────────────
def load_config() -> Dict:
    if not os.path.exists(CONFIG_PATH):
        default = {
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
            "news_headlines_count": 15,
            "spike_alert_pct": 3.0,
            "whale_volume_multiplier": 1.5,
            "signal_cycle_minutes": 60,
            "historical_baseline_profit": 1500,
            "optimizer_lookback_days": 14,
            "optimizer_simulation_loops": 15,
            "hf_dataset_repo": "sol-matrix-bot/cryptoai-state-data",
            "performance_metrics": {
                "total_signals_generated": 0,
                "successful_signals": 0,
                "failed_signals": 0,
                "current_win_rate_pct": 0.0
            }
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(default, f, indent=2)
        return default
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(config: Dict):
    with portalocker.Lock(CONFIG_PATH, timeout=5):
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)

def load_active_trade() -> Dict:
    try:
        with portalocker.Lock(ACTIVE_TRADE_PATH, timeout=5):
            with open(ACTIVE_TRADE_PATH, "r") as f:
                return json.load(f)
    except:
        return {}

def save_active_trade(data: Dict):
    with portalocker.Lock(ACTIVE_TRADE_PATH, timeout=5):
        with open(ACTIVE_TRADE_PATH, "w") as f:
            json.dump(data, f, indent=2)

# ── Save to Hugging Face ───────────────────────
def save_state_to_hf(config: Dict):
    try:
        api     = HfApi(token=HF_TOKEN)
        metrics = json.dumps(config.get("performance_metrics", {}), indent=2)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(metrics)
            temp_path = f.name
        api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo="performance_metrics.json",
            repo_id=config.get("hf_dataset_repo", "sol-matrix-bot/cryptoai-state-data"),
            repo_type="dataset",
            token=HF_TOKEN
        )
        os.unlink(temp_path)
        log.info("✅ State saved to Hugging Face")
    except Exception as e:
        log.error(f"HF save error: {e}")

# ── Telegram ───────────────────────────────────
def _sync_send_message(token: str, chat_id: str, text: str) -> Tuple[int, Dict]:
    url = f"https://api.telegram.org/bot{token.strip()}/sendMessage"
    resp = requests.post(url, json={"chat_id": str(chat_id).strip(), "text": text, "parse_mode": "HTML"}, timeout=15)
    return resp.status_code, resp.json()

def _sync_send_photo(token: str, chat_id: str, photo_path: str, caption: str) -> Tuple[int, Dict]:
    url = f"https://api.telegram.org/bot{token.strip()}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        resp = requests.post(url, data={"chat_id": str(chat_id).strip(), "caption": caption, "parse_mode": "HTML"}, files={"photo": photo}, timeout=20)
    return resp.status_code, resp.json()

async def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("⚠️ Telegram credentials missing!")
        return
    for attempt in range(3):
        try:
            status, data = await asyncio.to_thread(_sync_send_message, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, text)
            if status == 200:
                log.info("✅ Telegram message sent!")
                return
            else:
                log.warning(f"⚠️ Telegram status {status}: {data.get('description','')}")
        except Exception as e:
            log.warning(f"⚠️ Telegram attempt {attempt+1}: {e}")
            await asyncio.sleep(4)
    log.error("❌ Failed to send Telegram message")

async def send_telegram_photo(photo_path: str, caption: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("⚠️ Telegram credentials missing!")
        return
    if not os.path.exists(photo_path):
        await send_telegram_message(caption)
        return
    for attempt in range(3):
        try:
            status, data = await asyncio.to_thread(_sync_send_photo, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, photo_path, caption)
            if status == 200:
                log.info("✅ Telegram chart sent!")
                return
            else:
                log.warning(f"⚠️ Photo status {status}: {data.get('description','')}")
        except Exception as e:
            log.warning(f"⚠️ Photo attempt {attempt+1}: {e}")
            await asyncio.sleep(4)
    log.error("❌ Failed to send photo")

# ── Data Fetching ──────────────────────────────
COINGECKO_IDS = {"SOLUSDT": "solana", "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum"}
KRAKEN_PAIRS = {"SOLUSDT": "SOLUSD", "BTCUSDT": "XBTUSD", "ETHUSDT": "ETHUSD"}

async def fetch_current_price(symbol: str) -> Optional[float]:
    clean = symbol.replace("/","").upper()
    coin_id = COINGECKO_IDS.get(clean)
    if not coin_id:
        return None
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 200:
                    data  = await resp.json()
                    return float(data[coin_id]["usd"])
    except Exception as e:
        log.warning(f"⚠️ Price fetch error: {e}")
    return None

async def fetch_candles_coingecko(clean_symbol: str) -> Optional[pd.DataFrame]:
    coin_id = COINGECKO_IDS.get(clean_symbol)
    if not coin_id:
        return None
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=14"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    df   = pd.DataFrame(data, columns=["timestamp","open","high","low","close"])
                    df = df.astype({"open": float, "high": float, "low": float,  "close": float})
                    df["volume"]    = 1.0
                    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
                    return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        log.warning(f"⚠️ CoinGecko error: {e}")
    return None

async def fetch_candles_kraken(clean_symbol: str) -> Optional[pd.DataFrame]:
    kraken_pair = KRAKEN_PAIRS.get(clean_symbol)
    if not kraken_pair:
        return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.kraken.com/0/public/OHLC", params={"pair": kraken_pair, "interval": 60}, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                data = await resp.json()
                if data.get("error"): raise ValueError(f"Kraken: {data['error']}")
                key  = list(data["result"].keys())[0]
                rows = data["result"][key]
                df   = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","vwap","volume","count"])
                df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
                df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
                return df[["timestamp","open","high","low","close","volume"]].sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        log.warning(f"⚠️ Kraken error: {e}")
    return None

async def fetch_candles(symbol: str) -> Optional[pd.DataFrame]:
    clean = symbol.replace("/","").strip().upper()
    df    = await fetch_candles_coingecko(clean)
    if df is not None and len(df) >= 100:
        return df
    df = await fetch_candles_kraken(clean)
    return df

# ── Spike Detection ────────────────────────────
async def check_price_spikes(config: Dict):
    global last_prices
    pairs = config.get("trading_pair", ["SOL/USDT","BTC/USDT","ETH/USDT"])
    spike_pct = config.get("spike_alert_pct", 3.0)
    for pair in pairs:
        try:
            current_price = await fetch_current_price(pair)
            if current_price is None: continue
            clean = pair.replace("/","").upper()
            if clean in last_prices:
                old_price  = last_prices[clean]
                change_pct = ((current_price - old_price) / old_price * 100)
                if abs(change_pct) >= spike_pct:
                    direction = "📈 UP" if change_pct > 0 else "📉 DOWN"
                    await send_telegram_message(
                        f"⚡ <b>PRICE SPIKE ALERT!</b>\n\nPair: <b>{pair}</b>\nChange: <b>{change_pct:+.2f}%</b>\nCurrent: <code>${current_price:.4f}</code>"
                    )
            last_prices[clean] = current_price
        except Exception as e:
            log.warning(f"⚠️ Spike check error: {e}")

# ── Indicators & Pillars ───────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi"]         = RSIIndicator(df["close"], window=14).rsi()
    bb                = BollingerBands(df["close"], window=20)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_width"]    = bb.bollinger_wband()
    df["atr"]         = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    macd              = MACD(df["close"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    return df.dropna().reset_index(drop=True)

def check_market_regime(df: pd.DataFrame, config: Dict) -> bool:
    latest = df.iloc[-1]
    return not (latest["bb_width"] < 0.05 or latest["atr"] < config.get("atr_chop_threshold", 1.5))

def detect_liquidity_sweep(df: pd.DataFrame) -> bool:
    recent  = df.tail(5)
    vol_avg = df["volume"].rolling(20).mean().iloc[-1] or 1.0
    for _, c in recent.iterrows():
        body = abs(c["close"] - c["open"])
        wick = (c["open"] if c["close"] > c["open"] else c["close"]) - c["low"]
        if wick > body * 2 and c["close"] > c["open"] and c["volume"] >= vol_avg * 1.1:
            return True
    return False

async def check_whale_quick(df: pd.DataFrame, symbol: str):
    if df is None or len(df) < 20: return
    df = add_indicators(df)
    if detect_liquidity_sweep(df):
        await send_telegram_message(f"🐳 <b>WHALE SWEEP DETECTED!</b>\n\nPair: <b>{symbol}</b>\nPrice: <code>${df.iloc[-1]['close']:.4f}</code>")

async def fetch_news_sentiment() -> float:
    analyzer, headlines = SentimentIntensityAnalyzer(), []
    async with aiohttp.ClientSession() as session:
        for url in ["https://cointelegraph.com/rss", "https://decrypt.co/feed"]:
            try:
                async with session.get(url, timeout=8) as resp:
                    headlines.extend(re.findall(r'<title>(.*?)</title>', await resp.text())[2:12])
            except: pass
    if not headlines: return 0.0
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines[:15]]
    return sum(scores) / len(scores)

async def fetch_fear_greed() -> int:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.alternative.me/fng/", timeout=8) as resp:
                data = await resp.json()
                return int(data["data"][0]["value"])
    except:
        return 50

# ── Main Worker Execution Loop ─────────────────
async def main_loop():
    log.info("🚀 Starting Thread-Optimized Signal Engine...")
    config = load_config()
    await send_telegram_message("🤖 <b>Solana Matrix Bot Online on Railway!</b>")
    
    while True:
        try:
            config = load_config()
            await check_price_spikes(config)
            
            for pair in config.get("trading_pair", ["SOL/USDT"]):
                df = await fetch_candles(pair)
                if df is not None:
                    await check_whale_quick(df, pair)
            
            await asyncio.sleep(SPIKE_CHECK_SECONDS)
        except Exception as e:
            log.error(f"Error in execution core: {e}")
            await asyncio.sleep(60)

run_engine = lambda: asyncio.run(main_loop())
if __name__ == "__main__":
    threading.Thread(target=run_engine, daemon=True).start()
    while True: threading.Event().wait(3600)
