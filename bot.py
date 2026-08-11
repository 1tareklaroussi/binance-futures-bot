# ============================================================
# 🚀 CRYPTO SIGNAL BOT — GITHUB ACTIONS V3
# MACD + RSI REVERSAL + EMA200 + VOLUME + CANDLE CONFIRMATION
# Yahoo Finance OHLCV
# ============================================================

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ============================================================
# 🔐 TELEGRAM — GITHUB SECRETS
# ============================================================

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN is missing from GitHub Secrets")

if not TELEGRAM_CHAT_ID:
    raise RuntimeError("❌ TELEGRAM_CHAT_ID is missing from GitHub Secrets")

# ============================================================
# ⚙️ SETTINGS
# ============================================================

INTERVAL = "1h"
PERIOD = "30d"
MIN_RR = 1.5
ATR_SL_MULTIPLIER = 1.0
TP1_R = 1.5
TP2_R = 2.5

# ============================================================
# 📊 FILTER SETTINGS
# ============================================================
EMA_TREND_PERIOD = 200
VOLUME_PERIOD = 20
VOLUME_MULTIPLIER = 1.20
CANDLE_BODY_RATIO = 0.55
CANDLE_CLOSE_RATIO = 0.25

# ============================================================
# 🪙 SYMBOLS
# ============================================================

SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", 
    "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD", "MATIC-USD", "LTC-USD", 
    "SHIB-USD", "TRX-USD", "BCH-USD", "NEAR-USD", "UNI-USD", "ATOM-USD", 
    "XLM-USD", "XMR-USD", "ETC-USD", "ICP-USD", "FIL-USD", "HBAR-USD", 
    "VET-USD", "APT-USD", "OP-USD", "ARB-USD", "INJ-USD", "RNDR-USD", 
    "FTM-USD", "SUI-USD", "SEI-USD", "GALA-USD", "SAND-USD", "MANA-USD", 
    "AAVE-USD", "SNX-USD", "MKR-USD", "AXS-USD", "TIA-USD", "TAO-USD", 
    "KAS-USD", "STX-USD", "IMX-USD", "PEPE-USD", "WIF-USD", "BONK-USD", 
    "FLOKI-USD", "JUP-USD", "PYTH-USD", "RUNE-USD", "ALGO-USD", "EGLD-USD", 
    "QNT-USD", "EOS-USD", "XTZ-USD", "FLOW-USD", "THETA-USD", "CRV-USD", 
    "LDO-USD", "COMP-USD", "ZEC-USD", "DASH-USD"
]

# ============================================================
# 📝 LOGGING & HELPERS
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def telegram_symbol(symbol):
    return symbol.replace("-USD", "USDT")

def format_price(price):
    if price >= 1000: return f"{price:,.0f}"
    if price >= 100: return f"{price:,.2f}"
    if price >= 1: return f"{price:,.3f}"
    if price >= 0.01: return f"{price:,.5f}"
    return f"{price:,.8f}"

# ============================================================
# 📥 DATA FETCHING
# ============================================================

def get_data(symbol):
    try:
        df = yf.download(symbol, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            logging.warning(f"{symbol}: No data received")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        columns = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in columns):
            logging.warning(f"{symbol}: Missing required columns")
            return None

        df = df[columns].copy()
        for c in columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(inplace=True)

        if len(df) < 220:
            logging.warning(f"{symbol}: Not enough candles ({len(df)})")
            return None

        return df.iloc[:-1].copy()
    except Exception as e:
        logging.error(f"{symbol} data error: {e}")
        return None

# ============================================================
# 📊 INDICATORS
# ============================================================

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def MACD(series):
    ema12 = EMA(series, 12)
    ema26 = EMA(series, 26)
    macd = ema12 - ema26
    signal = EMA(macd, 9)
    histogram = macd - signal
    return macd, signal, histogram

def ATR(df, period=14):
    previous_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - previous_close).abs(),
        (df["Low"] - previous_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# ============================================================
# 🧠 ANALYSIS
# ============================================================

def analyze(symbol):
    df = get_data(symbol)
    if df is None or len(df) < 220: return None

    close = df["Close"]
    macd, macd_signal, histogram = MACD(close)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd, macd_signal, histogram
    df["RSI"] = RSI(close, 14)
    df["ATR"] = ATR(df, 14)
    df["EMA200"] = EMA(close, EMA_TREND_PERIOD)
    df["VOLUME_MA20"] = df["Volume"].rolling(VOLUME_PERIOD).mean()

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    bullish_cross = (prev["MACD"] <= prev["MACD_SIGNAL"] and curr["MACD"] > curr["MACD_SIGNAL"])
    bearish_cross = (prev["MACD"] >= prev["MACD_SIGNAL"] and curr["MACD"] < curr["MACD_SIGNAL"])

    recent_rsi = df["RSI"].iloc[-3:]
    rsi_oversold = (recent_rsi <= 30).any()
    rsi_overbought = (recent_rsi >= 70).any()

    bullish_trend = curr["Close"] > curr["EMA200"]
    bearish_trend = curr["Close"] < curr["EMA200"]

    volume_confirmed = curr["Volume"] > (curr["VOLUME_MA20"] * VOLUME_MULTIPLIER)

    candle_range = curr["High"] - curr["Low"]
    if candle_range <= 0: return None
    
    candle_body = abs(curr["Close"] - curr["Open"])
    body_ratio = candle_body / candle_range

    bullish_candle = (curr["Close"] > curr["Open"] and body_ratio >= CANDLE_BODY_RATIO and ((curr["High"] - curr["Close"]) / candle_range) <= CANDLE_CLOSE_RATIO)
    bearish_candle = (curr["Close"] < curr["Open"] and body_ratio >= CANDLE_BODY_RATIO and ((curr["Close"] - curr["Low"]) / candle_range) <= CANDLE_CLOSE_RATIO)

    direction = None
    reasons = []

    if bullish_cross and rsi_oversold and bullish_trend and volume_confirmed and bullish_candle:
        direction = "BUY"
        reasons.extend(["Bullish MACD Cross Up", f"Oversold RSI ({curr['RSI']:.1f})", "Price Above EMA200", "Volume Confirmed", "Bullish Candle Confirmed"])
    elif bearish_cross and rsi_overbought and bearish_trend and volume_confirmed and bearish_candle:
        direction = "SELL"
        reasons.extend(["Bearish MACD Cross Down", f"Overbought RSI ({curr['RSI']:.1f})", "Price Below EMA200", "Volume Confirmed", "Bearish Candle Confirmed"])
    else:
        return None

    price = float(curr["Close"])
    atr = float(curr["ATR"])
    if atr <= 0: return None
    risk = atr * ATR_SL_MULTIPLIER

    if direction == "BUY":
        sl, tp1, tp2 = price - risk, price + (risk * TP1_R), price + (risk * TP2_R)
        reward = tp1 - price
    else:
        sl, tp1, tp2 = price + risk, price - (risk * TP1_R), price - (risk * TP2_R)
        reward = price - tp1

    rr = reward / risk
    if rr < MIN_RR: return None

    return {
        "symbol": symbol, "direction": direction, "strength": 100,
        "entry": price, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr, "reasons": reasons
    }

# ============================================================
# 📲 TELEGRAM ALERT
# ============================================================

def send_signal(signal):
    symbol = telegram_symbol(signal["symbol"])
    icon = "🟢" if signal["direction"] == "BUY" else "🔴"

    message = (
        f"{icon} <b>NEW SIGNAL: {symbol}</b> {icon}\n\n"
        f"📊 <b>Direction:</b> {signal['direction']}\n"
        f"💪 <b>Strength:</b> {signal['strength']}%\n"
        f"🧠 <b>Strategy:</b> MACD + RSI Reversal + EMA200 + Volume + Candle\n"
        f"✅ <b>Reasons:</b> {', '.join(signal['reasons'])}\n\n"
        f"🎯 <b>Entry:</b> {format_price(signal['entry'])}\n"
        f"🛑 <b>Stop Loss:</b> {format_price(signal['sl'])}\n\n"
        f"💰 <b>Take Profit 1:</b> {format_price(signal['tp1'])}\n"
        f"🚀 <b>Take Profit 2:</b> {format_price(signal['tp2'])}\n\n"
        f"⚖️ <b>Risk/Reward:</b> 1:{signal['rr']:.2f}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=20)
        response.raise_for_status()
        print(f"📨 Telegram alert sent for {symbol}!")
        return True
    except Exception as e:
        print(f"❌ Telegram error for {symbol}: {e}")
        return False

# ============================================================
# 🔎 SCAN SYSTEM
# ============================================================

def scan():
    print("\n" + "=" * 60)
    print("🔎 NEW MARKET SCAN STARTED")
    print("=" * 60)

    signals = []

    for symbol in SYMBOLS:
        print(f"🔍 Analyzing {telegram_symbol(symbol)}...")
        try:
            signal = analyze(symbol)
            if signal:
                signals.append(signal)
                print(f"   ✅ {signal['direction']} | STRENGTH: {signal['strength']}% | RR: {signal['rr']:.2f}")
            else:
                print("   ⚪ No qualifying setup")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 🕒 تأخير زمني لمدة ثانيتين لتجنب الحظر من ياهو فاينانس
        time.sleep(2)

    if not signals:
        print("\n❌ No qualifying signals found in this scan cycle.")
        return

    print(f"\n🏆 FOUND {len(signals)} REVERSAL SIGNAL(S):")
    for sig in signals:
        print(f"📤 Sending alert for {telegram_symbol(sig['symbol'])}...")
        send_signal(sig)
        time.sleep(1)

# ============================================================
# 🚀 START BOT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 CRYPTO SIGNAL BOT V3 (GITHUB ACTIONS EDITION)")
    print(f"🪙 SYMBOLS: {len(SYMBOLS)}")
    print("=" * 60)
    scan()
    print("\n✅ Scan completed.")

