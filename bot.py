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
    raise RuntimeError(
        "❌ TELEGRAM_BOT_TOKEN is missing from GitHub Secrets"
    )

if not TELEGRAM_CHAT_ID:
    raise RuntimeError(
        "❌ TELEGRAM_CHAT_ID is missing from GitHub Secrets"
    )


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

# يجب أن يكون حجم التداول أكبر من متوسط
# آخر 20 شمعة بنسبة 20%
VOLUME_MULTIPLIER = 1.20

# الحد الأدنى لنسبة جسم الشمعة
CANDLE_BODY_RATIO = 0.55

# أقصى نسبة للمسافة بين الإغلاق والقمة/القاع
CANDLE_CLOSE_RATIO = 0.25


# ============================================================
# 🪙 SYMBOLS
# ============================================================

SYMBOLS = [

    # =========================
    # العملات الأصلية
    # =========================

    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "BNB-USD",
    "XRP-USD",
    "DOGE-USD",
    "ADA-USD",
    "AVAX-USD",
    "LINK-USD",
    "DOT-USD",
    "MATIC-USD",
    "LTC-USD",
    "SHIB-USD",
    "TRX-USD",
    "BCH-USD",
    "NEAR-USD",
    "UNI-USD",
    "ATOM-USD",
    "XLM-USD",
    "XMR-USD",
    "ETC-USD",
    "ICP-USD",
    "FIL-USD",
    "HBAR-USD",
    "VET-USD",
    "APT-USD",
    "OP-USD",
    "ARB-USD",
    "INJ-USD",
    "RNDR-USD",
    "FTM-USD",
    "SUI-USD",
    "SEI-USD",
    "GALA-USD",
    "SAND-USD",
    "MANA-USD",
    "AAVE-USD",
    "SNX-USD",
    "MKR-USD",
    "AXS-USD",
    "TIA-USD",
    "TAO-USD",
    "KAS-USD",
    "STX-USD",
    "IMX-USD",

    # =========================
    # ➕ 20 عملة إضافية
    # =========================

    "PEPE-USD",
    "WIF-USD",
    "BONK-USD",
    "FLOKI-USD",
    "JUP-USD",
    "PYTH-USD",
    "RUNE-USD",
    "ALGO-USD",
    "EGLD-USD",
    "QNT-USD",
    "EOS-USD",
    "XTZ-USD",
    "FLOW-USD",
    "THETA-USD",
    "CRV-USD",
    "LDO-USD",
    "MKR-USD",
    "COMP-USD",
    "ZEC-USD",
    "DASH-USD"
]


# ============================================================
# 📝 LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ============================================================
# SYMBOL FORMAT
# ============================================================

def telegram_symbol(symbol):

    return symbol.replace(
        "-USD",
        "USDT"
    )


# ============================================================
# 📥 DATA FETCHING
# ============================================================

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

            logging.warning(
                f"{symbol}: No data received"
            )

            return None


        # ====================================================
        # التعامل مع MultiIndex في yfinance
        # ====================================================

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = [
                c[0]
                for c in df.columns
            ]


        columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]


        if not all(
            c in df.columns
            for c in columns
        ):

            logging.warning(
                f"{symbol}: Missing required columns"
            )

            return None


        df = df[
            columns
        ].copy()


        # ====================================================
        # تحويل البيانات إلى أرقام
        # ====================================================

        for c in columns:

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )


        df.dropna(
            inplace=True
        )


        # نحتاج على الأقل 220 شمعة
        # EMA 200 + مساحة للمؤشرات
        if len(df) < 220:

            logging.warning(
                f"{symbol}: Not enough candles "
                f"({len(df)})"
            )

            return None


        # ====================================================
        # حذف الشمعة الحالية غير المكتملة
        # ====================================================

        df = df.iloc[
            :-1
        ].copy()


        return df


    except Exception as e:

        logging.error(
            f"{symbol} data error: {e}"
        )

        return None


# ============================================================
# 📊 EMA
# ============================================================

def EMA(
    series,
    period
):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


# ============================================================
# 📊 RSI
# ============================================================

def RSI(
    series,
    period=14
):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )


    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )


    return (
        100 -
        (
            100 /
            (1 + rs)
        )
    ).fillna(50)


# ============================================================
# 📊 MACD
# ============================================================

def MACD(series):

    ema12 = EMA(
        series,
        12
    )

    ema26 = EMA(
        series,
        26
    )


    macd = (
        ema12 -
        ema26
    )


    signal = EMA(
        macd,
        9
    )


    histogram = (
        macd -
        signal
    )


    return (
        macd,
        signal,
        histogram
    )


# ============================================================
# 📊 ATR
# ============================================================

def ATR(
    df,
    period=14
):

    previous_close = (
        df["Close"]
        .shift(1)
    )


    tr = pd.concat(
        [

            df["High"] -
            df["Low"],


            (
                df["High"] -
                previous_close
            ).abs(),


            (
                df["Low"] -
                previous_close
            ).abs()

        ],
        axis=1
    ).max(
        axis=1
    )


    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# 🧠 ANALYSIS
# ============================================================

def analyze(symbol):

    df = get_data(
        symbol
    )


    if df is None:

        return None


    if len(df) < 220:

        return None


    close = df[
        "Close"
    ]


    # ========================================================
    # المؤشرات الأصلية
    # ========================================================

    macd, macd_signal, histogram = MACD(
        close
    )


    df["MACD"] = macd

    df["MACD_SIGNAL"] = macd_signal

    df["MACD_HIST"] = histogram


    df["RSI"] = RSI(
        close,
        14
    )


    df["ATR"] = ATR(
        df,
        14
    )


    # ========================================================
    # 🆕 EMA 200 — فلتر الاتجاه
    # ========================================================

    df["EMA200"] = EMA(
        close,
        EMA_TREND_PERIOD
    )


    # ========================================================
    # 🆕 Volume Average
    # ========================================================

    df["VOLUME_MA20"] = (
        df["Volume"]
        .rolling(
            VOLUME_PERIOD
        )
        .mean()
    )


    # ========================================================
    # آخر شمعتين مغلقتين
    # ========================================================

    curr = df.iloc[-1]

    prev = df.iloc[-2]


    # ========================================================
    # 1️⃣ MACD CROSSOVER
    # ========================================================

    bullish_cross = (

        prev["MACD"]
        <=
        prev["MACD_SIGNAL"]

        and

        curr["MACD"]
        >
        curr["MACD_SIGNAL"]

    )


    bearish_cross = (

        prev["MACD"]
        >=
        prev["MACD_SIGNAL"]

        and

        curr["MACD"]
        <
        curr["MACD_SIGNAL"]

    )


    # ========================================================
    # 2️⃣ RSI EXTREME
    # ========================================================

    recent_rsi = (
        df["RSI"]
        .iloc[-3:]
    )


    rsi_oversold = (
        recent_rsi <= 30
    ).any()


    rsi_overbought = (
        recent_rsi >= 70
    ).any()


    # ========================================================
    # 3️⃣ 🆕 TREND FILTER
    # ========================================================

    bullish_trend = (

        curr["Close"]
        >
        curr["EMA200"]

    )


    bearish_trend = (

        curr["Close"]
        <
        curr["EMA200"]

    )


    # ========================================================
    # 4️⃣ 🆕 VOLUME FILTER
    # ========================================================

    volume_confirmed = (

        curr["Volume"]
        >
        (
            curr["VOLUME_MA20"]
            *
            VOLUME_MULTIPLIER
        )

    )


    # ========================================================
    # 5️⃣ 🆕 CANDLE CONFIRMATION
    # ========================================================

    candle_range = (

        curr["High"]
        -
        curr["Low"]

    )


    if candle_range <= 0:

        return None


    candle_body = abs(

        curr["Close"]
        -
        curr["Open"]

    )


    body_ratio = (

        candle_body /
        candle_range

    )


    # ========================================================
    # 🟢 Bullish candle
    # ========================================================

    bullish_candle = (

        curr["Close"]
        >
        curr["Open"]

        and

        body_ratio
        >=
        CANDLE_BODY_RATIO

        and

        (
            (
                curr["High"]
                -
                curr["Close"]
            )
            /
            candle_range
        )
        <=
        CANDLE_CLOSE_RATIO

    )


    # ========================================================
    # 🔴 Bearish candle
    # ========================================================

    bearish_candle = (

        curr["Close"]
        <
        curr["Open"]

        and

        body_ratio
        >=
        CANDLE_BODY_RATIO

        and

        (
            (
                curr["Close"]
                -
                curr["Low"]
            )
            /
            candle_range
        )
        <=
        CANDLE_CLOSE_RATIO

    )


    # ========================================================
    # 🎯 SIGNAL
    # ========================================================

    direction = None

    reasons = []


    # ========================================================
    # 🟢 BUY
    # ========================================================

    if (

        bullish_cross

        and

        rsi_oversold

        and

        bullish_trend

        and

        volume_confirmed

        and

        bullish_candle

    ):

        direction = "BUY"


        reasons.append(
            "Bullish MACD Cross Up"
        )


        reasons.append(
            f"Oversold RSI "
            f"({curr['RSI']:.1f})"
        )


        reasons.append(
            "Price Above EMA200"
        )


        reasons.append(
            "Volume Confirmed"
        )


        reasons.append(
            "Bullish Candle Confirmed"
        )


    # ========================================================
    # 🔴 SELL
    # ========================================================

    elif (

        bearish_cross

        and

        rsi_overbought

        and

        bearish_trend

        and

        volume_confirmed

        and

        bearish_candle

    ):

        direction = "SELL"


        reasons.append(
            "Bearish MACD Cross Down"
        )


        reasons.append(
            f"Overbought RSI "
            f"({curr['RSI']:.1f})"
        )


        reasons.append(
            "Price Below EMA200"
        )


        reasons.append(
            "Volume Confirmed"
        )


        reasons.append(
            "Bearish Candle Confirmed"
        )


    else:

        return None


    # ========================================================
    # 💰 RISK MANAGEMENT
    # ========================================================

    price = float(
        curr["Close"]
    )


    atr = float(
        curr["ATR"]
    )


    if atr <= 0:

        return None


    risk = (
        atr *
        ATR_SL_MULTIPLIER
    )


    # ========================================================
    # 🟢 BUY TARGETS
    # ========================================================

    if direction == "BUY":

        sl = (
            price -
            risk
        )


        tp1 = (
            price +
            (
                risk *
                TP1_R
            )
        )


        tp2 = (
            price +
            (
                risk *
                TP2_R
            )
        )


        reward = (
            tp1 -
            price
        )


    # ========================================================
    # 🔴 SELL TARGETS
    # ========================================================

    else:

        sl = (
            price +
            risk
        )


        tp1 = (
            price -
            (
                risk *
                TP1_R
            )
        )


        tp2 = (
            price -
            (
                risk *
                TP2_R
            )
        )


        reward = (
            price -
            tp1
        )


    # ========================================================
    # ⚖️ RISK / REWARD
    # ========================================================

    rr = (
        reward /
        risk
    )


    if rr < MIN_RR:

        return None


    # ========================================================
    # 💪 STRENGTH
    # ========================================================

    # جميع الفلاتر المطلوبة تحققت
    strength = 100


    return {

        "symbol":
            symbol,

        "direction":
            direction,

        "strength":
            strength,

        "entry":
            price,

        "sl":
            sl,

        "tp1":
            tp1,

        "tp2":
            tp2,

        "rr":
            rr,

        "reasons":
            reasons

    }


# ============================================================
# 💰 FORMAT PRICE
# ============================================================

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


# ============================================================
# 📲 TELEGRAM ALERT
# ============================================================

def send_signal(signal):

    symbol = telegram_symbol(
        signal["symbol"]
    )


    icon = (

        "🟢"

        if signal["direction"]
        == "BUY"

        else

        "🔴"

    )


    message = (

        f"{icon} "
        f"<b>NEW SIGNAL: "
        f"{symbol}</b> {icon}\n\n"


        f"📊 <b>Direction:</b> "
        f"{signal['direction']}\n"


        f"💪 <b>Strength:</b> "
        f"{signal['strength']}%\n"


        f"🧠 <b>Strategy:</b> "
        f"MACD + RSI Reversal + "
        f"EMA200 + Volume + Candle\n"


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

                "chat_id":
                    TELEGRAM_CHAT_ID,

                "text":
                    message,

                "parse_mode":
                    "HTML"

            },

            timeout=20

        )


        response.raise_for_status()


        print(
            f"📨 Telegram alert sent "
            f"for {symbol}!"
        )


        return True


    except Exception as e:

        print(
            f"❌ Telegram error "
            f"for {symbol}: {e}"
        )

        return False


# ============================================================
# 🔎 SCAN SYSTEM
# ============================================================

def scan():

    print(
        "\n" +
        "=" * 60
    )


    print(
        "🔎 NEW MARKET SCAN STARTED"
    )


    print(
        "=" * 60
    )


    signals = []


    for symbol in SYMBOLS:

        print(
            f"🔍 Analyzing "
            f"{telegram_symbol(symbol)}..."
        )


        try:

            signal = analyze(
                symbol
            )


            if signal:

                signals.append(
                    signal
                )


                print(

                    f"   ✅ "
                    f"{signal['direction']} | "
                    f"STRENGTH: "
                    f"{signal['strength']}% | "
                    f"RR: "
                    f"{signal['rr']:.2f}"

                )


            else:

                print(
                    "   ⚪ No qualifying setup"
                )


        except Exception as e:

            print(
                f"   ❌ Error: {e}"
            )


    # ========================================================
    # NO SIGNALS
    # ========================================================

    if not signals:

        print(
            "\n❌ No qualifying signals "
            "found in this scan cycle."
        )

        return


    # ========================================================
    # SEND SIGNALS
    # ========================================================

    print(

        f"\n🏆 FOUND "
        f"{len(signals)} "
        f"REVERSAL SIGNAL(S):"

    )


    for sig in signals:

        print(

            f"📤 Sending alert for "
            f"{telegram_symbol(sig['symbol'])}..."

        )


        send_signal(
            sig
        )


        # حماية Telegram من Rate Limit
        time.sleep(1)


# ============================================================
# 🚀 START BOT
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )


    print(
        "🚀 CRYPTO SIGNAL BOT V3"
    )


    print(
        "📊 MACD CROSSOVER + "
        "RSI REVERSAL"
    )


    print(
        "📈 EMA200 TREND FILTER"
    )


    print(
        "📊 VOLUME CONFIRMATION"
    )


    print(
        "🕯️ CANDLE CONFIRMATION"
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


    print(
        f"🪙 SYMBOLS: "
        f"{len(SYMBOLS)}"
    )


    print(
        "=" * 60
    )


    # تشغيل فحص واحد
    # GitHub Actions يعيد تشغيله كل ساعة

    scan()


    print(
        "\n✅ Scan completed."
    )
