# bot.py
# V5.1 — 1H Crypto Signal Bot
# MACD + Price Action + Trend + OB + FVG
# + Fibonacci/GZ + Volume Profile + Box + Risk
# + Telegram
#
# GitHub Secrets:
# TELEGRAM_BOT_TOKEN
# TELEGRAM_CHAT_ID
#
# Binance API Key is NOT required.

import os
import time
import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

BINANCE_BASE = "https://data-api.binance.vision"
TELEGRAM_URL = "https://api.telegram.org/bot{}/sendMessage"

TIMEFRAME = "1h"
CANDLE_LIMIT = 500

# Minimum score required to send a signal
MIN_SCORE = 70

# Strong signal
STRONG_SCORE = 80

# Maximum number of coins scanned
MAX_SYMBOLS = 80

# Minimum 24h USDT volume
MIN_24H_QUOTE_VOLUME = 10_000_000

# HTTP
REQUEST_TIMEOUT = 15
RETRIES = 3
REQUEST_DELAY = 0.15

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Trend
EMA_FAST = 20
EMA_SLOW = 50

# ATR / Risk
ATR_PERIOD = 14
SL_ATR_BUFFER = 0.20

RR1 = 2.0
RR2 = 3.0

# Analysis
SWING_LOOKBACK = 50
VP_LOOKBACK = 100
BOX_LOOKBACK = 30

# Duplicate protection
SENT_FILE = "sent_signals.txt"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

session = requests.Session()

session.headers.update({
    "User-Agent": "CryptoSignalBot/5.1"
})


# ============================================================
# HTTP
# ============================================================

def get_json(url, params=None):

    last_error = None

    for attempt in range(RETRIES):

        try:

            response = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            return response.json()

        except Exception as exc:

            last_error = exc

            logging.warning(
                f"Request failed "
                f"({attempt + 1}/{RETRIES}): {exc}"
            )

            time.sleep(
                1.5 * (attempt + 1)
            )

    raise RuntimeError(
        f"HTTP error: {last_error}"
    )


# ============================================================
# BINANCE SYMBOLS
# ============================================================

def get_symbols():

    data = get_json(
        f"{BINANCE_BASE}/api/v3/exchangeInfo"
    )

    symbols = []

    for item in data.get("symbols", []):

        if (
            item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get(
                "isSpotTradingAllowed",
                True
            )
        ):

            symbols.append(
                item["symbol"]
            )

    return symbols


# ============================================================
# 24H VOLUME
# ============================================================

def get_24h_tickers():

    data = get_json(
        f"{BINANCE_BASE}/api/v3/ticker/24hr"
    )

    result = {}

    for item in data:

        try:

            result[item["symbol"]] = {
                "quote_volume":
                    float(item["quoteVolume"]),

                "price":
                    float(item["lastPrice"]),

                "change_pct":
                    float(item["priceChangePercent"])
            }

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            continue

    return result


# ============================================================
# OHLCV
# ============================================================

def get_klines(
    symbol,
    limit=CANDLE_LIMIT
):

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

    df = pd.DataFrame(
        data,
        columns=columns
    )

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

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df = df.reset_index(
        drop=True
    )

    # ========================================================
    # IMPORTANT:
    # Remove current unfinished candle.
    # ========================================================

    if len(df) > 0:

        now = pd.Timestamp.now(
            tz="UTC"
        )

        if (
            df.iloc[-1]["close_time"]
            > now
        ):

            df = df.iloc[:-1].copy()

    return df.reset_index(
        drop=True
    )


# ============================================================
# EMA
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
# ATR
# ============================================================

def ATR(
    df,
    period=ATR_PERIOD
):

    previous_close = (
        df["close"].shift(1)
    )

    tr = pd.concat(
        [
            df["high"] - df["low"],

            (
                df["high"]
                - previous_close
            ).abs(),

            (
                df["low"]
                - previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# MACD
# ============================================================

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

    histogram = (
        macd_line
        - signal_line
    )

    return (
        macd_line,
        signal_line,
        histogram
    )


# ============================================================
# MACD CROSSOVER
# PRIMARY SIGNAL
# ============================================================

def macd_signal(df):

    if len(df) < 50:

        return None, 0

    macd_line, signal_line, histogram = MACD(
        df
    )

    previous_macd = (
        macd_line.iloc[-2]
    )

    previous_signal = (
        signal_line.iloc[-2]
    )

    current_macd = (
        macd_line.iloc[-1]
    )

    current_signal = (
        signal_line.iloc[-1]
    )

    # Bullish crossover
    bullish_cross = (
        previous_macd
        <= previous_signal
        and
        current_macd
        > current_signal
    )

    # Bearish crossover
    bearish_cross = (
        previous_macd
        >= previous_signal
        and
        current_macd
        < current_signal
    )

    if bullish_cross:

        return "LONG", 25

    if bearish_cross:

        return "SHORT", 25

    return None, 0


# ============================================================
# CANDLE HELPERS
# ============================================================

def candle_body(c):

    return abs(
        c["close"]
        - c["open"]
    )


def candle_range(c):

    return max(
        c["high"]
        - c["low"],
        1e-12
    )


# ============================================================
# ENGULFING
# ============================================================

def bullish_engulfing(
    previous,
    current
):

    return (
        previous["close"]
        < previous["open"]
        and
        current["close"]
        > current["open"]
        and
        current["open"]
        <= previous["close"]
        and
        current["close"]
        >= previous["open"]
    )


def bearish_engulfing(
    previous,
    current
):

    return (
        previous["close"]
        > previous["open"]
        and
        current["close"]
        < current["open"]
        and
        current["open"]
        >= previous["close"]
        and
        current["close"]
        <= previous["open"]
    )


# ============================================================
# PIN BAR
# ============================================================

def pin_bar(
    candle,
    bullish=True
):

    body = candle_body(
        candle
    )

    rng = candle_range(
        candle
    )

    upper_wick = (
        candle["high"]
        - max(
            candle["open"],
            candle["close"]
        )
    )

    lower_wick = (
        min(
            candle["open"],
            candle["close"]
        )
        - candle["low"]
    )

    if body / rng > 0.35:

        return False

    if bullish:

        return (
            lower_wick >= body * 2
            and
            lower_wick
            >= upper_wick * 1.5
        )

    return (
        upper_wick >= body * 2
        and
        upper_wick
        >= lower_wick * 1.5
    )


# ============================================================
# PRICE ACTION
# ============================================================

def price_action_signal(df):

    if len(df) < 10:

        return None, 0

    previous = df.iloc[-2]
    current = df.iloc[-1]

    bullish = (
        bullish_engulfing(
            previous,
            current
        )
        or
        pin_bar(
            current,
            True
        )
    )

    bearish = (
        bearish_engulfing(
            previous,
            current
        )
        or
        pin_bar(
            current,
            False
        )
    )

    recent_high = (
        df["high"]
        .iloc[-7:-1]
        .max()
    )

    recent_low = (
        df["low"]
        .iloc[-7:-1]
        .min()
    )

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
        EMA_FAST
    )

    ema50 = EMA(
        df["close"],
        EMA_SLOW
    )

    price = (
        df["close"].iloc[-1]
    )

    bullish = (
        price > ema20.iloc[-1]
        and
        ema20.iloc[-1]
        > ema50.iloc[-1]
        and
        ema20.iloc[-1]
        > ema20.iloc[-6]
    )

    bearish = (
        price < ema20.iloc[-1]
        and
        ema20.iloc[-1]
        < ema50.iloc[-1]
        and
        ema20.iloc[-1]
        < ema20.iloc[-6]
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

    x = np.arange(
        len(highs)
    )

    high_slope = np.polyfit(
        x,
        highs.values,
        1
    )[0]

    low_slope = np.polyfit(
        x,
        lows.values,
        1
    )[0]

    if (
        high_slope > 0
        and
        low_slope > 0
    ):

        return "LONG", 5

    if (
        high_slope < 0
        and
        low_slope < 0
    ):

        return "SHORT", 5

    return None, 0


# ============================================================
# ORDER BLOCK
# ============================================================

def order_block_signal(df):

    if len(df) < 30:

        return None, 0, None

    recent = df.iloc[-15:]

    atr_value = ATR(
        df
    ).iloc[-1]

    if (
        not np.isfinite(atr_value)
        or
        atr_value <= 0
    ):

        return None, 0, None

    current = df.iloc[-1]

    # Search backwards for the last opposite candle
    # before an impulsive move.
    for i in range(
        len(recent) - 3,
        0,
        -1
    ):

        candle = recent.iloc[i]

        future = recent.iloc[
            i + 1:
        ]

        if len(future) == 0:

            continue

        future_high = (
            future["high"].max()
        )

        future_low = (
            future["low"].min()
        )

        upward_move = (
            future_high
            - candle["high"]
        )

        downward_move = (
            candle["low"]
            - future_low
        )

        # Bullish OB
        if candle["close"] < candle["open"]:

            if upward_move >= atr_value * 1.2:

                zone = (
                    float(candle["low"]),
                    float(candle["high"])
                )

                touched = (
                    current["low"]
                    <= zone[1]
                    and
                    current["high"]
                    >= zone[0]
                )

                if touched:

                    return (
                        "LONG",
                        10,
                        zone
                    )

        # Bearish OB
        if candle["close"] > candle["open"]:

            if downward_move >= atr_value * 1.2:

                zone = (
                    float(candle["low"]),
                    float(candle["high"])
                )

                touched = (
                    current["high"]
                    >= zone[0]
                    and
                    current["low"]
                    <= zone[1]
                )

                if touched:

                    return (
                        "SHORT",
                        10,
                        zone
                    )

    return None, 0, None


# ============================================================
# FAIR VALUE GAP
# ============================================================

def fvg_signal(df):

    if len(df) < 10:

        return None, 0, None

    a = df.iloc[-3]
    b = df.iloc[-2]
    c = df.iloc[-1]

    # Bullish FVG:
    # first candle high < third candle low
    if a["high"] < c["low"]:

        zone = (
            float(a["high"]),
            float(c["low"])
        )

        return (
            "LONG",
            10,
            zone
        )

    # Bearish FVG:
    # first candle low > third candle high
    if a["low"] > c["high"]:

        zone = (
            float(c["high"]),
            float(a["low"])
        )

        return (
            "SHORT",
            10,
            zone
        )

    return None, 0, None


# ============================================================
# FIBONACCI + GOLDEN ZONE
# ============================================================

def fibonacci_signal(df):

    if len(df) < SWING_LOOKBACK:

        return None, 0, None

    window = df.iloc[
        -SWING_LOOKBACK:
    ]

    swing_high = float(
        window["high"].max()
    )

    swing_low = float(
        window["low"].min()
    )

    current = float(
        df["close"].iloc[-1]
    )

    diff = (
        swing_high
        - swing_low
    )

    if (
        diff <= 0
        or
        not np.isfinite(diff)
    ):

        return None, 0, None

    # IMPORTANT:
    # Use explicit numeric calculations.
    # This fixes the previous KeyError: '0.5'.

    fib_382 = (
        swing_high
        - diff * 0.382
    )

    fib_500 = (
        swing_high
        - diff * 0.500
    )

    fib_618 = (
        swing_high
        - diff * 0.618
    )

    fib_650 = (
        swing_high
        - diff * 0.650
    )

    fib_786 = (
        swing_high
        - diff * 0.786
    )

    levels = {
        "0.382": fib_382,
        "0.500": fib_500,
        "0.618": fib_618,
        "0.650": fib_650,
        "0.786": fib_786
    }

    # 2.5% tolerance of the swing range
    tolerance = diff * 0.025

    # ========================================================
    # GOLDEN ZONE
    # ========================================================

    golden_zone = (
        abs(
            current - fib_618
        ) <= tolerance
        or
        abs(
            current - fib_650
        ) <= tolerance
    )

    if golden_zone:

        if current >= fib_500:

            return (
                "LONG",
                10,
                levels
            )

        return (
            "SHORT",
            10,
            levels
        )

    # ========================================================
    # 0.382
    # ========================================================

    if abs(
        current - fib_382
    ) <= tolerance:

        if current >= fib_500:

            return (
                "LONG",
                8,
                levels
            )

        return (
            "SHORT",
            8,
            levels
        )

    # ========================================================
    # 0.500
    # ========================================================

    if abs(
        current - fib_500
    ) <= tolerance:

        if current >= fib_500:

            return (
                "LONG",
                8,
                levels
            )

        return (
            "SHORT",
            8,
            levels
        )

    # ========================================================
    # 0.786
    # ========================================================

    if abs(
        current - fib_786
    ) <= tolerance:

        if current >= fib_500:

            return (
                "LONG",
                8,
                levels
            )

        return (
            "SHORT",
            8,
            levels
        )

    return None, 0, levels


# ============================================================
# VOLUME PROFILE
# ============================================================

def volume_profile_signal(df):

    if len(df) < VP_LOOKBACK:

        return None, 0, None

    data = df.iloc[
        -VP_LOOKBACK:
    ].copy()

    price_min = float(
        data["low"].min()
    )

    price_max = float(
        data["high"].max()
    )

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

        typical_price = (
            row["high"]
            + row["low"]
            + row["close"]
        ) / 3

        index = (
            np.searchsorted(
                bins,
                typical_price,
                side="right"
            )
            - 1
        )

        if (
            0 <= index
            < len(volume_profile)
        ):

            volume_profile[index] += (
                row["volume"]
            )

    poc_index = int(
        np.argmax(
            volume_profile
        )
    )

    poc = (
        bins[poc_index]
        + bins[poc_index + 1]
    ) / 2

    current = float(
        df["close"].iloc[-1]
    )

    tolerance = (
        price_max
        - price_min
    ) * 0.03

    if (
        current > poc
        and
        abs(current - poc)
        <= tolerance
    ):

        return (
            "LONG",
            10,
            poc
        )

    if (
        current < poc
        and
        abs(current - poc)
        <= tolerance
    ):

        return (
            "SHORT",
            10,
            poc
        )

    return None, 0, poc


# ============================================================
# BOX / RANGE BREAKOUT
# ============================================================

def box_signal(df):

    if len(df) < BOX_LOOKBACK:

        return None, 0, None

    data = df.iloc[
        -BOX_LOOKBACK:
    ]

    previous = data.iloc[:-1]

    resistance = float(
        previous["high"].max()
    )

    support = float(
        previous["low"].min()
    )

    current = float(
        df["close"].iloc[-1]
    )

    if current > resistance:

        return (
            "LONG",
            5,
            (support, resistance)
        )

    if current < support:

        return (
            "SHORT",
            5,
            (support, resistance)
        )

    return (
        None,
        0,
        (support, resistance)
    )


# ============================================================
# RISK MANAGEMENT
# ============================================================

def calculate_risk(
    df,
    direction,
    ob_zone=None
):

    current = float(
        df["close"].iloc[-1]
    )

    atr_value = float(
        ATR(df).iloc[-1]
    )

    if (
        not np.isfinite(atr_value)
        or
        atr_value <= 0
    ):

        return None

    recent = df.iloc[-20:]

    swing_low = float(
        recent["low"].min()
    )

    swing_high = float(
        recent["high"].max()
    )

    if direction == "LONG":

        stop_candidates = [
            swing_low
            - atr_value * SL_ATR_BUFFER
        ]

        if ob_zone is not None:

            stop_candidates.append(
                ob_zone[0]
                - atr_value * SL_ATR_BUFFER
            )

        sl = min(
            stop_candidates
        )

        risk = (
            current - sl
        )

        if risk <= 0:

            return None

        tp1 = (
            current
            + risk * RR1
        )

        tp2 = (
            current
            + risk * RR2
        )

    else:

        stop_candidates = [
            swing_high
            + atr_value * SL_ATR_BUFFER
        ]

        if ob_zone is not None:

            stop_candidates.append(
                ob_zone[1]
                + atr_value * SL_ATR_BUFFER
            )

        sl = max(
            stop_candidates
        )

        risk = (
            sl - current
        )

        if risk <= 0:

            return None

        tp1 = (
            current
            - risk * RR1
        )

        tp2 = (
            current
            - risk * RR2
        )

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
# ANALYSIS / SCORE
# ============================================================

def analyze(
    symbol,
    df
):

    if len(df) < 120:

        return None

    components = {}

    # Primary signal
    macd_dir, macd_points = (
        macd_signal(df)
    )

    # Do not waste analysis if there
    # is no MACD crossover.
    if macd_dir is None:

        return None

    trend_dir, trend_points = (
        trend_signal(df)
    )

    trendline_dir, trendline_points = (
        trendline_signal(df)
    )

    pa_dir, pa_points = (
        price_action_signal(df)
    )

    ob_dir, ob_points, ob_zone = (
        order_block_signal(df)
    )

    fvg_dir, fvg_points, fvg_zone = (
        fvg_signal(df)
    )

    fib_dir, fib_points, fib_levels = (
        fibonacci_signal(df)
    )

    vp_dir, vp_points, poc = (
        volume_profile_signal(df)
    )

    box_dir, box_points, box_zone = (
        box_signal(df)
    )

    components["MACD"] = (
        macd_dir,
        macd_points
    )

    components["Trend"] = (
        trend_dir,
        trend_points
    )

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

    # ========================================================
    # Score only the MACD direction.
    # ========================================================

    long_score = 0
    short_score = 0

    for direction, points in (
        components.values()
    ):

        if direction == "LONG":

            long_score += points

        elif direction == "SHORT":

            short_score += points

    if macd_dir == "LONG":

        direction = "LONG"
        score = long_score

    else:

        direction = "SHORT"
        score = short_score

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

    if not token:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing."
        )

    if not chat_id:

        raise RuntimeError(
            "TELEGRAM_CHAT_ID is missing."
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


# ============================================================
# PRICE FORMAT
# ============================================================

def fmt_price(value):

    value = float(value)

    if value >= 1000:

        return f"{value:,.2f}"

    if value >= 1:

        return f"{value:.4f}"

    if value >= 0.01:

        return f"{value:.6f}"

    return f"{value:.8f}"


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def format_signal(signal):

    symbol = signal["symbol"]
    direction = signal["direction"]
    score = signal["score"]
    risk = signal["risk"]

    if direction == "LONG":

        emoji = "🟢 LONG"

    else:

        emoji = "🔴 SHORT"

    if score >= STRONG_SCORE:

        strength = "🔥 STRONG"

    else:

        strength = "✅ VALID"

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

        result = signal[
            "components"
        ][name]

        direction_result = result[0]
        points = result[1]

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

            f"🕐 Candle: "
            f"{signal['candle_time']}",

            "",

            "⚠️ Signal only — "
            "not financial advice.",

            "━━━━━━━━━━━━━━━━━━━━"

        ]
    )

    return "\n".join(lines)


# ============================================================
# DUPLICATE PROTECTION
# ============================================================

def load_sent():

    if not os.path.exists(
        SENT_FILE
    ):

        return set()

    try:

        with open(
            SENT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return {
                line.strip()
                for line in file
                if line.strip()
            }

    except Exception:

        return set()


def save_sent(key):

    with open(
        SENT_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            key + "\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    logging.info(
        "🚀 Starting V5.1 1H scanner"
    )

    sent = load_sent()

    symbols = get_symbols()

    tickers = get_24h_tickers()

    candidates = []

    for symbol in symbols:

        info = tickers.get(
            symbol
        )

        if info is None:

            continue

        if (
            info["quote_volume"]
            < MIN_24H_QUOTE_VOLUME
        ):

            continue

        candidates.append(
            (
                symbol,
                info["quote_volume"]
            )
        )

    candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    candidates = candidates[
        :MAX_SYMBOLS
    ]

    logging.info(
        f"🔎 Scanning "
        f"{len(candidates)} "
        f"liquid USDT pairs"
    )

    signals_sent = 0

    for symbol, _ in candidates:

        try:

            df = get_klines(
                symbol
            )

            if len(df) < 120:

                logging.warning(
                    f"{symbol}: "
                    f"not enough candles"
                )

                continue

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
                    f"⏭️ Duplicate: "
                    f"{symbol}"
                )

                continue

            message = format_signal(
                signal
            )

            send_telegram(
                message
            )

            save_sent(
                key
            )

            sent.add(
                key
            )

            signals_sent += 1

            logging.info(
                f"📩 SENT "
                f"{symbol} "
                f"{signal['direction']} "
                f"Score={signal['score']}"
            )

        except Exception as exc:

            # IMPORTANT:
            # One bad symbol must never stop
            # the entire scanner.
            logging.error(
                f"{symbol}: {exc}"
            )

        time.sleep(
            REQUEST_DELAY
        )

    logging.info(
        f"✅ Scan finished. "
        f"Signals sent: {signals_sent}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
