# ============================================================
# 🚀 CRYPTO SIGNAL BOT — GITHUB ACTIONS V2
# MACD Crossover + RSI Extreme Reversal
# Yahoo Finance OHLCV
# ============================================================

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
import yfinance as yf


# =========================
# 🔐 TELEGRAM SECRETS
# =========================

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN is missing from GitHub Secrets")

if not TELEGRAM_CHAT_ID:
    raise RuntimeError("❌ TELEGRAM_CHAT_ID is missing from GitHub Secrets")


# =========================
# ⚙️ SETTINGS
# =========================

INTERVAL = "1h"
PERIOD = "30d"

MIN_RR = 1.5

ATR_SL_MULTIPLIER = 1.0
TP1_R = 1.5
TP2_R = 2.5


# =========================
# 📋 SYMBOLS
# =========================

SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "DOGE-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
    "MATIC-USD", "LTC-USD", "SHIB-USD", "TRX-USD", "BCH-USD",
    "NEAR-USD", "UNI-USD", "ATOM-USD", "XLM-USD", "XMR-USD",
    "ETC-USD", "ICP-USD", "FIL-USD", "HBAR-USD", "VET-USD",
    "APT-USD", "OP-USD", "ARB-USD", "INJ-USD", "RNDR-USD",
    "FTM-USD", "SUI-USD", "SEI-USD", "GALA-USD", "SAND-USD",
    "MANA-USD", "AAVE-USD", "SNX-USD", "MKR-USD", "AXS-USD",
    "TIA-USD", "TAO-USD", "KAS-USD", "STX-USD", "IMX-USD"
]


# =========================
# 📝 LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# =========================
# SYMBOL FORMAT
# =========================

def telegram_symbol(symbol):
    return symbol.replace("-USD", "USDT")


# =========================
# DATA FETCHING
# =========================

def get_data(symbol):
    try:

        df = yf.download(
            symbol,
            period=PERIOD,
            interval=INTERVAL,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if df is None or df.empty:
            logging.warning(f"{symbol}: No data received")
            return None

        # التعامل مع MultiIndex الخاص بـ yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(c in df.columns for c in columns):
            logging.warning(f"{symbol}: Missing required columns")
            return None

        df = df[columns].copy()

        for c in columns:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

        df.dropna(inplace=True)

        if len(df) < 220:
            logging.warning(
                f"{symbol}: Not enough candles ({len(df)})"
            )
            return None

        # حذف الشمعة الحالية غير المكتملة
        df = df.iloc[:-1].copy()

        return df

    except Exception as e:
        logging.error(f"{symbol} data error: {e}")
        return None


# =========================
# 📊 INDICATORS
# =========================

def EMA(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def RSI(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    return (
        100 - (100 / (1 + rs))
    ).fillna(50)


def MACD(series):

    ema12 = EMA(series, 12)

    ema26 = EMA(series, 26)

    macd = ema12 - ema26

    signal = EMA(macd, 9)

    histogram = macd - signal

    return macd, signal, histogram


def ATR(df, period=14):

    previous_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],

            (
                df["High"] - previous_close
            ).abs(),

            (
                df["Low"] - previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# =========================
# 🧠 ANALYSIS
# =========================

def analyze(symbol):

    df = get_data(symbol)

    if df is None or len(df) < 50:
        return None

    close = df["Close"]

    # MACD
    macd, macd_signal, histogram = MACD(close)

    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd_signal

    # RSI
    df["RSI"] = RSI(close, 14)

    # ATR
    df["ATR"] = ATR(df, 14)

    # آخر شمعتين مغلقتين
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # =========================
    # MACD CROSS
    # =========================

    bullish_cross = (
        prev["MACD"] <= prev["MACD_SIGNAL"]
        and
        curr["MACD"] > curr["MACD_SIGNAL"]
    )

    bearish_cross = (
        prev["MACD"] >= prev["MACD_SIGNAL"]
        and
        curr["MACD"] < curr["MACD_SIGNAL"]
    )

    # =========================
    # RSI EXTREME
    # =========================

    recent_rsi = df["RSI"].iloc[-3:]

    rsi_oversold = (
        recent_rsi <= 30
    ).any()

    rsi_overbought = (
        recent_rsi >= 70
    ).any()

    direction = None

    reasons = []

    # =========================
    # 🟢 BUY
    # =========================

    if bullish_cross and rsi_oversold:

        direction = "BUY"

        reasons.append(
            f"Bullish Reversal | "
            f"MACD Cross Up | "
            f"Oversold RSI ({curr['RSI']:.1f})"
        )

    # =========================
    # 🔴 SELL
    # =========================

    elif bearish_cross and rsi_overbought:

        direction = "SELL"

        reasons.append(
            f"Bearish Reversal | "
            f"MACD Cross Down | "
            f"Overbought RSI ({curr['RSI']:.1f})"
        )

    else:
        return None

    # =========================
    # RISK MANAGEMENT
    # =========================

    price = float(curr["Close"])

    atr = float(curr["ATR"])

    if atr <= 0:
        return None

    risk = atr * ATR_SL_MULTIPLIER

    if direction == "BUY":

        sl = price - risk

        tp1 = price + (
            risk * TP1_R
        )

        tp2 = price + (
            risk * TP2_R
        )

        reward = tp1 - price

    else:

        sl = price + risk

        tp1 = price - (
            risk * TP1_R
        )

        tp2 = price - (
            risk * TP2_R
        )

        reward = price - tp1

    rr = reward / risk

    if rr < MIN_RR:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "strength": 100,
        "entry": price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "reasons": reasons
    }


# =========================
# 💰 PRICE FORMAT
# =========================

def format_price(price):

    if price >= 1000:
        return f"{price:,.0f}"

    if price >= 100:
        return f"{price:,.2f}"

    if price >= 1:
        return f"{price:,.3f}"

    if price >= 0.01:
        return f"{price:,.5f}"

    return f"{price:,.8f}"


# =========================
# 📲 TELEGRAM
# =========================

def send_signal(signal):

    symbol = telegram_symbol(
        signal["symbol"]
    )

    icon = (
        "🟢"
        if signal["direction"] == "BUY"
        else "🔴"
    )

    message = (
        f"{icon} <b>NEW SIGNAL: {symbol}</b> {icon}\n\n"

        f"📊 <b>Direction:</b> "
        f"{signal['direction']}\n"

        f"💪 <b>Strength:</b> "
        f"{signal['strength']}%\n"

        f"🧠 <b>Strategy:</b> "
        f"MACD Cross + RSI Extreme Reversal\n"

        f"✅ <b>Reasons:</b> "
        f"{', '.join(signal['reasons'])}\n\n"

        f"🎯 <b>Entry:</b> "
        f"{format_price(signal['entry'])}\n"

        f"🛑 <b>Stop Loss:</b> "
        f"{format_price(signal['sl'])}\n\n"

        f"💰 <b>Take Profit 1:</b> "
        f"{format_price(signal['tp1'])}\n"

        f"🚀 <b>Take Profit 2:</b> "
        f"{format_price(signal['tp2'])}\n\n"

        f"⚖️ <b>Risk/Reward:</b> "
        f"1:{signal['rr']:.2f}"
    )

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    try:

        response = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            },
            timeout=20
        )

        response.raise_for_status()

        print(
            f"📨 Telegram alert sent for {symbol}!"
        )

        return True

    except Exception as e:

        print(
            f"❌ Telegram error for {symbol}: {e}"
        )

        return False


# =========================
# 🔎 SCAN
# =========================

def scan():

    print("\n" + "=" * 60)
    print("🔎 NEW MARKET SCAN STARTED")
    print("=" * 60)

    signals = []

    for symbol in SYMBOLS:

        print(
            f"🔍 Analyzing "
            f"{telegram_symbol(symbol)}..."
        )

        try:

            signal = analyze(symbol)

            if signal:

                signals.append(signal)

                print(
                    f"   ✅ "
                    f"{signal['direction']} | "
                    f"STRENGTH: 100% | "
                    f"RR: {signal['rr']:.2f}"
                )

            else:

                print(
                    "   ⚪ No qualifying setup"
                )

        except Exception as e:

            print(
                f"   ❌ Error: {e}"
            )

    # =========================
    # NO SIGNALS
    # =========================

    if not signals:

        print(
            "\n❌ No qualifying signals "
            "found in this scan."
        )

        return

    # =========================
    # SEND SIGNALS
    # =========================

    print(
        f"\n🏆 FOUND "
        f"{len(signals)} SIGNAL(S):"
    )

    for sig in signals:

        print(
            f"📤 Sending alert for "
            f"{telegram_symbol(sig['symbol'])}..."
        )

        send_signal(sig)

        # Telegram rate-limit protection
        time.sleep(1)


# =========================
# 🚀 START
# =========================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "🚀 CRYPTO SIGNAL BOT V2"
    )

    print(
        "📊 MACD CROSSOVER + "
        "RSI REVERSAL"
    )

    print(
        "📡 DATA: Yahoo Finance"
    )

    print(
        "⏱️ TIMEFRAME: 1H"
    )

    print(
        f"🎯 MIN RR: {MIN_RR}"
    )

    print("=" * 60)

    # تشغيل فحص واحد فقط
    # GitHub Actions سيعيد تشغيله تلقائياً
    scan()

    print("\n✅ Scan completed.")
