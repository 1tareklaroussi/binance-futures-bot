# bot.py
# V5 - 1H Crypto Signal Bot
# MACD + Price Action + Trend + OB + FVG + Fibonacci/GZ
# + Volume Profile + Box + Risk Management + Telegram
#
# Required GitHub Secrets:
# TELEGRAM_BOT_TOKEN
# TELEGRAM_CHAT_ID

import os
import time
import logging
import requests
import numpy as np
import pandas as pd

BINANCE_BASE = "https://data-api.binance.vision"
TELEGRAM_URL = "https://api.telegram.org/bot{}/sendMessage"

TIMEFRAME = "1h"
CANDLE_LIMIT = 500

MIN_SCORE = 70
STRONG_SCORE = 80

ATR_PERIOD = 14
RR1 = 2.0
RR2 = 3.0
SL_ATR_BUFFER = 0.20

MAX_SYMBOLS = 80
MIN_24H_QUOTE_VOLUME = 10_000_000
REQUEST_TIMEOUT = 15

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

session = requests.Session()
session.headers.update({"User-Agent": "CryptoSignalBot/5.0"})


# ============================================================
# HTTP
# ============================================================

def get_json(url, params=None, retries=3):
    last_error = None

    for attempt in range(retries):
        try:
            response = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            last_error = e
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"HTTP error: {last_error}")


# ============================================================
# BINANCE DATA
# ============================================================

def get_symbols():
    data = get_json(
        f"{BINANCE_BASE}/api/v3/exchangeInfo"
    )

    symbols = []

    for s in data.get("symbols", []):

        if (
            s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and s.get("isSpotTradingAllowed", True)
        ):
            symbols.append(s["symbol"])

    return symbols


def get_24h_tickers():

    data = get_json(
        f"{BINANCE_BASE}/api/v3/ticker/24hr"
    )

    result = {}

    for x in data:

        try:
            result[x["symbol"]] = {
                "quote_volume": float(x["quoteVolume"]),
                "price": float(x["lastPrice"]),
                "change_pct": float(x["priceChangePercent"])
            }

        except Exception:
            continue

    return result


def get_klines(symbol, limit=CANDLE_LIMIT):

    data = get_json(
        f"{BINANCE_BASE}/api/v3/klines",
        {
            "symbol": symbol,
            "interval": TIMEFRAME,
            "limit": limit
        }
    )

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore"
    ]

    df = pd.DataFrame(data, columns=columns)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_volume",
        "taker_buy_quote_volume"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True
    )

    # لا نستخدم الشمعة الحالية غير المغلقة
    now = pd.Timestamp.now(tz="UTC")

    if (
        len(df) > 0
        and df.iloc[-1]["close_time"] > now
    ):
        df = df.iloc[:-1].copy()

    return df.reset_index(drop=True)


# ============================================================
# INDICATORS
# ============================================================

def EMA(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def ATR(df, period=ATR_PERIOD):

    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


def MACD(df):

    fast = EMA(
        df["close"],
        MACD_FAST
    )

    slow = EMA(
        df["close"],
        MACD_SLOW
    )

    macd_line = fast - slow

    signal_line = EMA(
        macd_line,
        MACD_SIGNAL
    )

    histogram = macd_line - signal_line

    return (
        macd_line,
        signal_line,
        histogram
    )


# ============================================================
# MACD CROSSOVER
# ============================================================

def macd_signal(df):

    macd_line, signal_line, histogram = MACD(df)

    if len(df) < 50:
        return None, 0

    previous_macd = macd_line.iloc[-2]
    previous_signal = signal_line.iloc[-2]

    current_macd = macd_line.iloc[-1]
    current_signal = signal_line.iloc[-1]

    bullish_cross = (
        previous_macd <= previous_signal
        and current_macd > current_signal
    )

    bearish_cross = (
        previous_macd >= previous_signal
        and current_macd < current_signal
    )

    if bullish_cross:
        return "LONG", 25

    if bearish_cross:
        return "SHORT", 25

    return None, 0


# ============================================================
# PRICE ACTION
# ============================================================

def candle_body(c):
    return abs(c["close"] - c["open"])


def candle_range(c):
    return max(
        c["high"] - c["low"],
        1e-12
    )


def bullish_engulfing(previous, current):

    return (
        previous["close"] < previous["open"]
        and current["close"] > current["open"]
        and current["open"] <= previous["close"]
        and current["close"] >= previous["open"]
    )


def bearish_engulfing(previous, current):

    return (
        previous["close"] > previous["open"]
        and current["close"] < current["open"]
        and current["open"] >= previous["close"]
        and current["close"] <= previous["open"]
    )


def pin_bar(candle, bullish=True):

    body = candle_body(candle)
    rng = candle_range(candle)

    upper_wick = (
        candle["high"]
        - max(candle["open"], candle["close"])
    )

    lower_wick = (
        min(candle["open"], candle["close"])
        - candle["low"]
    )

    if body / rng > 0.35:
        return False

    if bullish:
        return (
            lower_wick >= body * 2
            and lower_wick >= upper_wick * 1.5
        )

    return (
        upper_wick >= body * 2
        and upper_wick >= lower_wick * 1.5
    )


def price_action_signal(df):

    if len(df) < 10:
        return None, 0

    previous = df.iloc[-2]
    current = df.iloc[-1]

    bullish = (
        bullish_engulfing(previous, current)
        or pin_bar(current, True)
    )

    bearish = (
        bearish_engulfing(previous, current)
        or pin_bar(current, False)
    )

    recent_high = df["high"].iloc[-7:-1].max()
    recent_low = df["low"].iloc[-7:-1].min()

    if current["close"] > recent_high:
        bullish = True

    if current["close"] < recent_low:
        bearish = True

    if bullish and not bearish:
        return "LONG", 15

    if bearish and not bullish:
        return "SHORT", 15

    return None, 0


# ============================================================
# TREND
# ============================================================

def trend_signal(df):

    if len(df) < 80:
        return None, 0

    ema20 = EMA(
        df["close"],
        20
    )

    ema50 = EMA(
        df["close"],
        50
    )

    price = df["close"].iloc[-1]

    bullish = (
        price > ema20.iloc[-1]
        and ema20.iloc[-1] > ema50.iloc[-1]
        and ema20.iloc[-1] > ema20.iloc[-6]
    )

    bearish = (
        price < ema20.iloc[-1]
        and ema20.iloc[-1] < ema50.iloc[-1]
        and ema20.iloc[-1] < ema20.iloc[-6]
    )

    if bullish:
        return "LONG", 15

    if bearish:
        return "SHORT", 15

    return None, 0


# ============================================================
# TREND LINE
# ============================================================

def trendline_signal(df):

    if len(df) < 30:
        return None, 0

    highs = df["high"].iloc[-20:]
    lows = df["low"].iloc[-20:]

    high_slope = np.polyfit(
        np.arange(len(highs)),
        highs.values,
        1
    )[0]

    low_slope = np.polyfit(
        np.arange(len(lows)),
        lows.values,
        1
    )[0]

    if high_slope > 0 and low_slope > 0:
        return "LONG", 5

    if high_slope < 0 and low_slope < 0:
        return "SHORT", 5

    return None, 0


# ============================================================
# ORDER BLOCK
# ============================================================

def order_block_signal(df):

    if len(df) < 20:
        return None, 0, None

    recent = df.iloc[-12:]

    current = df.iloc[-1]

    for i in range(len(recent) - 3, 0, -1):

        candle = recent.iloc[i]
        next_candles = recent.iloc[i + 1:]

        move_up = (
            next_candles["close"].max()
            - candle["high"]
        )

        move_down = (
            candle["low"]
            - next_candles["close"].min()
        )

        # Bullish OB = last bearish candle before strong rise
        if candle["close"] < candle["open"]:

            if move_up > ATR(df).iloc[-1] * 1.2:

                if (
                    current["low"] <= candle["high"]
                    and current["high"] >= candle["low"]
                ):
                    return (
                        "LONG",
                        10,
                        (candle["low"], candle["high"])
                    )

        # Bearish OB = last bullish candle before strong fall
        if candle["close"] > candle["open"]:

            if move_down > ATR(df).iloc[-1] * 1.2:

                if (
                    current["high"] >= candle["low"]
                    and current["low"] <= candle["high"]
                ):
                    return (
                        "SHORT",
                        10,
                        (candle["low"], candle["high"])
                    )

    return None, 0, None


# ============================================================
# FVG
# ============================================================

def fvg_signal(df):

    if len(df) < 10:
        return None, 0, None

    # Three-candle FVG
    a = df.iloc[-3]
    b = df.iloc[-2]
    c = df.iloc[-1]

    # Bullish FVG: candle A high < candle C low
    if a["high"] < c["low"]:

        zone = (
            a["high"],
            c["low"]
        )

        if (
            c["low"] <= zone[1]
            and c["high"] >= zone[0]
        ):
            return "LONG", 10, zone

    # Bearish FVG: candle A low > candle C high
    if a["low"] > c["high"]:

        zone = (
            c["high"],
            a["low"]
        )

        if (
            c["high"] >= zone[0]
            and c["low"] <= zone[1]
        ):
            return "SHORT", 10, zone

    return None, 0, None


# ============================================================
# FIBONACCI / GOLDEN ZONE
# ============================================================

def fibonacci_signal(df):

    if len(df) < 50:
        return None, 0, None

    window = df.iloc[-50:]

    swing_high = window["high"].max()
    swing_low = window["low"].min()

    current = df["close"].iloc[-1]

    diff = swing_high - swing_low

    if diff <= 0:
        return None, 0, None

    # Retracement levels
    levels = {
        "0.382": swing_high - diff * 0.382,
        "0.500": swing_high - diff * 0.500,
        "0.618": swing_high - diff * 0.618,
        "0.650": swing_high - diff * 0.650,
        "0.786": swing_high - diff * 0.786
    }

    tolerance = diff * 0.025

    # Golden zone
    if (
        abs(current - levels["0.618"]) <= tolerance
        or abs(current - levels["0.650"]) <= tolerance
    ):
        if current > swing_low + diff * 0.5:
            return "LONG", 10, levels

        return "SHORT", 10, levels

    for name, level in levels.items():

        if abs(current - level) <= tolerance:

            if current > levels["0.5"]:
                return "LONG", 8, levels

            return "SHORT", 8, levels

    return None, 0, levels


# ============================================================
# VOLUME PROFILE
# ============================================================

def volume_profile_signal(df):

    if len(df) < 100:
        return None, 0, None

    data = df.iloc[-100:].copy()

    price_min = data["low"].min()
    price_max = data["high"].max()

    if price_max <= price_min:
        return None, 0, None

    bins = np.linspace(
        price_min,
        price_max,
        25
    )

    volume_profile = np.zeros(
        len(bins) - 1
    )

    for _, row in data.iterrows():

        price = (
            row["high"]
            + row["low"]
            + row["close"]
        ) / 3

        index = np.searchsorted(
            bins,
            price,
            side="right"
        ) - 1

        if 0 <= index < len(volume_profile):
            volume_profile[index] += row["volume"]

    poc_index = int(
        np.argmax(volume_profile)
    )

    poc = (
        bins[poc_index]
        + bins[poc_index + 1]
    ) / 2

    current = df["close"].iloc[-1]

    tolerance = (
        price_max - price_min
    ) * 0.03

    if current > poc and abs(current - poc) <= tolerance:
        return "LONG", 10, poc

    if current < poc and abs(current - poc) <= tolerance:
        return "SHORT", 10, poc

    return None, 0, poc


# ============================================================
# BOX / SUPPORT RESISTANCE
# ============================================================

def box_signal(df):

    if len(df) < 30:
        return None, 0, None

    data = df.iloc[-30:]

    resistance = data["high"].iloc[:-1].max()
    support = data["low"].iloc[:-1].min()

    current = df["close"].iloc[-1]

    atr_value = ATR(df).iloc[-1]

    if pd.isna(atr_value):
        return None, 0, None

    # Breakout above box
    if current > resistance:
        return "LONG", 5, (support, resistance)

    # Breakout below box
    if current < support:
        return "SHORT", 5, (support, resistance)

    return None, 0, (support, resistance)


# ============================================================
# RISK / SL / TP
# ============================================================

def calculate_risk(df, direction, ob_zone=None):

    current = float(df["close"].iloc[-1])
    atr_value = float(ATR(df).iloc[-1])

    if not np.isfinite(atr_value) or atr_value <= 0:
        return None

    recent = df.iloc[-20:]

    swing_low = float(
        recent["low"].min()
    )

    swing_high = float(
        recent["high"].max()
    )

    if direction == "LONG":

        candidates = [
            swing_low - atr_value * SL_ATR_BUFFER
        ]

        if ob_zone:
            candidates.append(
                ob_zone[0] - atr_value * SL_ATR_BUFFER
            )

        sl = min(candidates)

        risk = current - sl

        if risk <= 0:
            return None

        tp1 = current + risk * RR1
        tp2 = current + risk * RR2

    else:

        candidates = [
            swing_high + atr_value * SL_ATR_BUFFER
        ]

        if ob_zone:
            candidates.append(
                ob_zone[1] + atr_value * SL_ATR_BUFFER
            )

        sl = max(candidates)

        risk = sl - current

        if risk <= 0:
            return None

        tp1 = current - risk * RR1
        tp2 = current - risk * RR2

    return {
        "entry": current,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk": risk,
        "rr1": RR1,
        "rr2": RR2
    }


# ============================================================
# SCORE ENGINE
# ============================================================

def analyze(symbol, df):

    if len(df) < 120:
        return None

    components = {}

    macd_dir, macd_points = macd_signal(df)
    trend_dir, trend_points = trend_signal(df)
    trendline_dir, trendline_points = trendline_signal(df)
    pa_dir, pa_points = price_action_signal(df)

    ob_dir, ob_points, ob_zone = order_block_signal(df)
    fvg_dir, fvg_points, fvg_zone = fvg_signal(df)
    fib_dir, fib_points, fib_levels = fibonacci_signal(df)
    vp_dir, vp_points, poc = volume_profile_signal(df)
    box_dir, box_points, box_zone = box_signal(df)

    components["MACD"] = (macd_dir, macd_points)
    components["Trend"] = (trend_dir, trend_points)
    components["Trend Line"] = (
        trendline_dir,
        trendline_points
    )
    components["Price Action"] = (
        pa_dir,
        pa_points
    )
    components["Order Block"] = (
        ob_dir,
        ob_points
    )
    components["FVG"] = (
        fvg_dir,
        fvg_points
    )
    components["Fibonacci"] = (
        fib_dir,
        fib_points
    )
    components["Volume Profile"] = (
        vp_dir,
        vp_points
    )
    components["Box"] = (
        box_dir,
        box_points
    )

    # MACD must be the primary trigger.
    if macd_dir is None:
        return None

    long_score = 0
    short_score = 0

    for name, (direction, points) in components.items():

        if direction == "LONG":
            long_score += points

        elif direction == "SHORT":
            short_score += points

    if macd_dir == "LONG":
        score = long_score
        direction = "LONG"
    else:
        score = short_score
        direction = "SHORT"

    if score < MIN_SCORE:
        return None

    risk = calculate_risk(
        df,
        direction,
        ob_zone
    )

    if risk is None:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "score": score,
        "components": components,
        "risk": risk,
        "poc": poc,
        "ob_zone": ob_zone,
        "fvg_zone": fvg_zone,
        "fib_levels": fib_levels,
        "candle_time": df.iloc[-1]["close_time"]
    }


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing."
        )

    response = requests.post(
        TELEGRAM_URL.format(token),
        data={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        },
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()


def fmt_price(value):

    value = float(value)

    if value >= 1000:
        return f"{value:,.2f}"

    if value >= 1:
        return f"{value:.4f}"

    if value >= 0.01:
        return f"{value:.6f}"

    return f"{value:.8f}"


def format_signal(signal):

    symbol = signal["symbol"]
    direction = signal["direction"]
    score = signal["score"]
    risk = signal["risk"]

    emoji = (
        "🟢 LONG"
        if direction == "LONG"
        else "🔴 SHORT"
    )

    strength = (
        "🔥 STRONG"
        if score >= STRONG_SCORE
        else "✅ VALID"
    )

    lines = [
        "━━━━━━━━━━━━━━━━━━━━",
        f"{emoji} | {strength}",
        "📊 <b>1H CRYPTO SIGNAL</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"🪙 <b>{symbol}</b>",
        f"⭐ Score: <b>{score}/100</b>",
        "",
        "📌 <b>ENTRY / RISK</b>",
        f"Entry: {fmt_price(risk['entry'])}",
        f"SL: {fmt_price(risk['sl'])}",
        f"TP1: {fmt_price(risk['tp1'])}",
        f"TP2: {fmt_price(risk['tp2'])}",
        f"R:R: 1:{RR1:g} / 1:{RR2:g}",
        "",
        "🔍 <b>CONFIRMATIONS</b>"
    ]

    names = [
        "MACD",
        "Trend",
        "Trend Line",
        "Price Action",
        "Order Block",
        "FVG",
        "Fibonacci",
        "Volume Profile",
        "Box"
    ]

    for name in names:

        direction_result, points = signal[
            "components"
        ][name]

        if direction_result == direction:
            mark = "✅"
        elif direction_result is None:
            mark = "⚪"
        else:
            mark = "❌"

        lines.append(
            f"{mark} {name}: {points} pts"
        )

    lines.extend(
        [
            "",
            f"🕐 Candle: {signal['candle_time']}",
            "",
            "⚠️ Signal only — not financial advice.",
            "━━━━━━━━━━━━━━━━━━━━"
        ]
    )

    return "\n".join(lines)


# ============================================================
# DUPLICATE PROTECTION
# ============================================================

SENT_FILE = "sent_signals.txt"


def load_sent():

    if not os.path.exists(SENT_FILE):
        return set()

    with open(
        SENT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        return {
            line.strip()
            for line in f
            if line.strip()
        }


def save_sent(key):

    with open(
        SENT_FILE,
        "a",
        encoding="utf-8"
    ) as f:
        f.write(key + "\n")


# ============================================================
# MAIN SCANNER
# ============================================================

def main():

    logging.info("🚀 Starting V5 1H scanner")

    sent = load_sent()

    symbols = get_symbols()
    tickers = get_24h_tickers()

    # Select the most liquid USDT pairs.
    candidates = []

    for symbol in symbols:

        info = tickers.get(symbol)

        if not info:
            continue

        if info["quote_volume"] < MIN_24H_QUOTE_VOLUME:
            continue

        candidates.append(
            (
                symbol,
                info["quote_volume"]
            )
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    candidates = candidates[:MAX_SYMBOLS]

    logging.info(
        f"🔎 Scanning {len(candidates)} liquid USDT pairs"
    )

    found = []

    for symbol, _ in candidates:

        try:

            df = get_klines(symbol)

            signal = analyze(
                symbol,
                df
            )

            if signal is None:
                continue

            key = (
                f"{symbol}|"
                f"{signal['direction']}|"
                f"{signal['candle_time']}"
            )

            if key in sent:
                logging.info(
                    f"⏭️ Duplicate: {symbol}"
                )
                continue

            message = format_signal(
                signal
            )

            send_telegram(message)

            save_sent(key)
            sent.add(key)

            found.append(signal)

            logging.info(
                f"📩 SENT {symbol} "
                f"{signal['direction']} "
                f"Score={signal['score']}"
            )

        except Exception as e:

            logging.error(
                f"{symbol}: {e}"
            )

        # Stay gentle with public API.
        time.sleep(0.15)

    logging.info(
        f"✅ Scan finished. Signals sent: {len(found)}"
    )


if __name__ == "__main__":
    main()
