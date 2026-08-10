# ============================================================
# 🚀 CRYPTO SIGNAL BOT — COLAB V2 (SAFE REVERSAL STRATEGY)
# Yahoo Finance OHLCV
# Focus: MACD Crossover + RSI Extreme Reversal (Overbought/Oversold)
# ============================================================

!pip -q install yfinance pandas numpy requests

import time
import logging
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# =========================
# 🔐 TELEGRAM (HARDCODED)
# =========================

TELEGRAM_BOT_TOKEN = "8990510095:AAH8Gdtp3xjsCN2vR00GJ1NvEUQj0GLsXnQ"
TELEGRAM_CHAT_ID = "7776631198"


# =========================
# ⚙️ SETTINGS
# =========================

INTERVAL = "1h"
PERIOD = "30d"

MIN_RR = 1.5

ATR_SL_MULTIPLIER = 1.0
TP1_R = 1.5
TP2_R = 2.5

SCAN_INTERVAL = 3600 # الفحص كل ساعة (3600 ثانية)

# قائمة العملات الرقمية
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
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        columns = ["Open", "High", "Low", "Close", "Volume"]

        if not all(c in df.columns for c in columns):
            return None

        df = df[columns].copy()

        for c in columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df.dropna(inplace=True)

        if len(df) < 220:
            return None

        # حذف الشمعة الحالية غير المكتملة لضمان دقة الإشارة
        return df.iloc[:-1].copy()

    except Exception as e:
        logging.error(f"{symbol} data error: {e}")
        return None


# =========================
# INDICATORS
# =========================

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


# =========================
# ANALYSIS (SAFE REVERSAL STRATEGY)
# =========================

def analyze(symbol):
    df = get_data(symbol)

    if df is None or len(df) < 50:
        return None

    close = df["Close"]

    # حساب المؤشرات الفنية
    macd, macd_signal, histogram = MACD(close)
    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd_signal
    df["RSI"] = RSI(close, 14)
    df["ATR"] = ATR(df, 14)

    # شمعة الإغلاق الأخيرة والتي قبلها
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # 1️⃣ شرط تقاطع MACD الخطي الصريح (Crossover)
    bullish_cross = (prev["MACD"] <= prev["MACD_SIGNAL"]) and (curr["MACD"] > curr["MACD_SIGNAL"])
    bearish_cross = (prev["MACD"] >= prev["MACD_SIGNAL"]) and (curr["MACD"] < curr["MACD_SIGNAL"])

    # 2️⃣ شرط التشبع الشرائي والبيعي لـ RSI في آخر 3 شمعات
    recent_rsi = df["RSI"].iloc[-3:]
    rsi_oversold = (recent_rsi <= 30).any()   # حالة نزول شديد -> توقع صعود
    rsi_overbought = (recent_rsi >= 70).any() # حالة صعود شديد -> توقع هبوط

    direction = None
    reasons = []

    # 🟢 شراء: العملة في حالة نزول (RSI <= 30) + تقاطع MACD للأعلى
    if bullish_cross and rsi_oversold:
        direction = "BUY"
        reasons.append(f"Bullish Reversal | MACD Cross Up | Oversold RSI ({curr['RSI']:.1f})")

    # 🔴 بيع: العملة في حالة صعود (RSI >= 70) + تقاطع MACD للأسفل
    elif bearish_cross and rsi_overbought:
        direction = "SELL"
        reasons.append(f"Bearish Reversal | MACD Cross Down | Overbought RSI ({curr['RSI']:.1f})")

    else:
        return None  # عدم استيفاء الشروط

    # ========================================================
    # إدارة المخاطر والأهداف
    # ========================================================
    price = float(curr["Close"])
    atr = float(curr["ATR"])
    if atr <= 0: 
        return None

    risk = atr * ATR_SL_MULTIPLIER

    if direction == "BUY":
        sl = price - risk
        tp1 = price + (risk * TP1_R)
        tp2 = price + (risk * TP2_R)
        reward = tp1 - price
    else:
        sl = price + risk
        tp1 = price - (risk * TP1_R)
        tp2 = price - (risk * TP2_R)
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
# FORMAT PRICE
# =========================

def format_price(price):
    if price >= 1000: return f"{price:,.0f}"
    if price >= 100: return f"{price:,.2f}"
    if price >= 1: return f"{price:,.3f}"
    if price >= 0.01: return f"{price:,.5f}"
    return f"{price:,.8f}"


# =========================
# TELEGRAM ALERTS
# =========================

def send_signal(signal):
    symbol = telegram_symbol(signal["symbol"])
    
    icon = "🟢" if signal['direction'] == "BUY" else "🔴"
    
    message = (
        f"{icon} <b>NEW SIGNAL: {symbol}</b> {icon}\n\n"
        f"📊 <b>Direction:</b> {signal['direction']}\n"
        f"💪 <b>Strength:</b> {signal['strength']}%\n"
        f"🧠 <b>Strategy:</b> MACD Cross + RSI Extreme Reversal\n"
        f"✅ <b>Reasons:</b> {', '.join(signal['reasons'])}\n\n"
        f"🎯 <b>Entry:</b> {format_price(signal['entry'])}\n"
        f"🛑 <b>Stop Loss:</b> {format_price(signal['sl'])}\n\n"
        f"💰 <b>Take Profit 1:</b> {format_price(signal['tp1'])}\n"
        f"🚀 <b>Take Profit 2:</b> {format_price(signal['tp2'])}\n\n"
        f"⚖️ <b>Risk/Reward:</b> 1:{signal['rr']:.2f}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

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
        print(f"📨 Telegram alert sent for {symbol}!")
        return True
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False


# =========================
# SCAN SYSTEM
# =========================

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
                print(f"   ✅ {signal['direction']} | STRENGTH: 100% | RR: {signal['rr']:.2f}")
            else:
                print("   ⚪ No MACD cross + RSI extreme setup")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    if not signals:
        print("\n❌ No qualifying signals found in this scan cycle.")
        return

    # إرسال جميع الإشارات المطابقة للشروط
    print(f"\n🏆 FOUND {len(signals)} REVERSAL SIGNAL(S):")
    for sig in signals:
        print(f"📤 Sending alert for {telegram_symbol(sig['symbol'])}...")
        send_signal(sig)
        time.sleep(1)  # فاصل زمن لتجنب الحظر من التليجرام


# =========================
# START BOT
# =========================

print("=" * 60)
print("🚀 CRYPTO SIGNAL BOT V2 (MACD CROSSOVER + RSI REVERSAL)")
print("📡 DATA: Yahoo Finance | TIMEFRAME: 1H")
print(f"🎯 MIN RR: {MIN_RR}")
print(f"🤖 Connected to Telegram ID: {TELEGRAM_CHAT_ID}")
print("=" * 60)

# تشغيل الفحص الأول فوراً
scan()
print("\n⏳ Next scan in 60 minutes...")

# حلقة العمل المستمرة
while True:
    try:
        time.sleep(SCAN_INTERVAL)
        scan()
        print("\n⏳ Next scan in 60 minutes...")
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped manually.")
        break
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        time.sleep(60)

