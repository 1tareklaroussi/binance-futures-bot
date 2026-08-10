import os
import time
import logging
import requests
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BYBIT_BASE = "https://api.bybit.com"

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

INTERVAL = "60"          # 1H
KLINE_LIMIT = 250

MIN_SIGNAL_STRENGTH = 70

# عدد العملات التي سيتم تحليلها
MAX_SYMBOLS = 40

# تأخير بسيط بين الطلبات
REQUEST_DELAY = 0.15

# عدد محاولات الاتصال
MAX_RETRIES = 5


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

session = requests.Session()

# منع تكرار نفس الإشارة أثناء تشغيل البرنامج
last_signals = {}


# ============================================================
# HTTP REQUEST
# ============================================================

def api_get(path, params=None):

    for attempt in range(MAX_RETRIES):

        try:

            response = session.get(
                BYBIT_BASE + path,
                params=params,
                timeout=15
            )

            # Rate limit
            if response.status_code == 429:

                wait = min(
                    2 ** attempt,
                    30
                )

                logging.warning(
                    f"Bybit 429 - waiting {wait}s"
                )

                time.sleep(wait)
                continue

            # Server errors
            if response.status_code >= 500:

                wait = min(
                    2 ** attempt,
                    30
                )

                logging.warning(
                    f"Bybit HTTP {response.status_code} "
                    f"- waiting {wait}s"
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

            data = response.json()

            # Bybit API-level error
            if data.get("retCode", 0) != 0:

                logging.warning(
                    f"Bybit API error: "
                    f"{data.get('retCode')} "
                    f"{data.get('retMsg')}"
                )

                return None

            time.sleep(REQUEST_DELAY)

            return data

        except requests.RequestException as e:

            wait = min(
                2 ** attempt,
                30
            )

            logging.warning(
                f"Request error: {e} "
                f"| retry in {wait}s"
            )

            time.sleep(wait)

    return None


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:

        response = session.post(
            url,
            data=payload,
            timeout=15
        )

        response.raise_for_status()

        logging.info(
            "Telegram signal sent successfully"
        )

        return True

    except requests.RequestException as e:

        logging.error(
            f"Telegram error: {e}"
        )

        return False


# ============================================================
# GET LINEAR USDT SYMBOLS
# ============================================================

def get_symbols():

    data = api_get(
        "/v5/market/instruments-info",
        {
            "category": "linear",
            "limit": 1000
        }
    )

    if not data:
        return []

    result = []

    items = data.get(
        "result",
        {}
    ).get(
        "list",
        []
    )

    for item in items:

        if (
            item.get("status") == "Trading"
            and item.get("contractType") == "LinearPerpetual"
            and item.get("settleCoin") == "USDT"
        ):

            result.append(
                item["symbol"]
            )

    return result


# ============================================================
# GET 24H TICKERS
# ============================================================

def get_tickers():

    data = api_get(
        "/v5/market/tickers",
        {
            "category": "linear"
        }
    )

    if not data:
        return {}

    result = {}

    items = data.get(
        "result",
        {}
    ).get(
        "list",
        []
    )

    for item in items:

        symbol = item.get("symbol")

        if not symbol:
            continue

        try:

            turnover = float(
                item.get(
                    "turnover24h",
                    0
                )
            )

            result[symbol] = turnover

        except (ValueError, TypeError):

            result[symbol] = 0

    return result


# ============================================================
# GET KLINES
# ============================================================

def get_klines(symbol):

    data = api_get(
        "/v5/market/kline",
        {
            "category": "linear",
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": KLINE_LIMIT
        }
    )

    if not data:
        return None

    rows = data.get(
        "result",
        {}
    ).get(
        "list",
        []
    )

    if not rows:
        return None

    # Bybit returns newest first.
    rows = list(reversed(rows))

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover"
    ]

    df = pd.DataFrame(
        rows,
        columns=columns
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    df = df.dropna()

    return df


# ============================================================
# EMA
# ============================================================

def calculate_ema(
    series,
    period
):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
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

    rsi = (
        100 -
        (100 / (1 + rs))
    )

    return rsi.fillna(50)


# ============================================================
# MACD
# ============================================================

def calculate_macd(
    series,
    fast=12,
    slow=26,
    signal=9
):

    fast_ema = calculate_ema(
        series,
        fast
    )

    slow_ema = calculate_ema(
        series,
        slow
    )

    macd = fast_ema - slow_ema

    signal_line = calculate_ema(
        macd,
        signal
    )

    histogram = (
        macd -
        signal_line
    )

    return (
        macd,
        signal_line,
        histogram
    )


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    df,
    period=14
):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high -
        previous_close
    ).abs()

    tr3 = (
        low -
        previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    return atr


# ============================================================
# ANALYZE SYMBOL
# ============================================================

def analyze_symbol(symbol):

    df = get_klines(symbol)

    if df is None:
        return None

    if len(df) < 220:
        return None

    # --------------------------------------------------------
    # Ignore currently forming candle
    # --------------------------------------------------------

    df = df.iloc[:-1].copy()

    if len(df) < 200:
        return None

    close = df["close"]

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    df["ema50"] = calculate_ema(
        close,
        50
    )

    df["ema200"] = calculate_ema(
        close,
        200
    )

    (
        df["macd"],
        df["macd_signal"],
        df["macd_hist"]
    ) = calculate_macd(
        close
    )

    df["rsi"] = calculate_rsi(
        close
    )

    df["atr"] = calculate_atr(
        df
    )

    # --------------------------------------------------------
    # Current closed candle
    # --------------------------------------------------------

    current = df.iloc[-1]
    previous = df.iloc[-2]

    price = float(
        current["close"]
    )

    ema50 = float(
        current["ema50"]
    )

    ema200 = float(
        current["ema200"]
    )

    macd = float(
        current["macd"]
    )

    macd_signal = float(
        current["macd_signal"]
    )

    macd_hist = float(
        current["macd_hist"]
    )

    previous_hist = float(
        previous["macd_hist"]
    )

    rsi = float(
        current["rsi"]
    )

    atr = float(
        current["atr"]
    )

    if not np.isfinite(atr) or atr <= 0:
        return None

    # ========================================================
    # SIGNAL SCORE
    # ========================================================

    buy_score = 0
    sell_score = 0

    # ========================================================
    # EMA TREND = 25
    # ========================================================

    if ema50 > ema200:

        buy_score += 25

    elif ema50 < ema200:

        sell_score += 25

    # ========================================================
    # MACD = 25
    # ========================================================

    if (
        macd > macd_signal
        and macd_hist > 0
    ):

        buy_score += 25

    elif (
        macd < macd_signal
        and macd_hist < 0
    ):

        sell_score += 25

    # ========================================================
    # RSI = 20
    # ========================================================

    if 50 <= rsi <= 68:

        buy_score += 20

    elif 32 <= rsi < 50:

        sell_score += 20

    elif rsi > 68:

        buy_score += 10

    elif rsi < 32:

        sell_score += 10

    # ========================================================
    # VOLUME = 15
    # ========================================================

    avg_volume = (
        df["volume"]
        .iloc[-21:-1]
        .mean()
    )

    current_volume = float(
        current["volume"]
    )

    if avg_volume > 0:

        volume_ratio = (
            current_volume /
            avg_volume
        )

    else:

        volume_ratio = 0

    candle_direction = (
        float(current["close"]) >
        float(current["open"])
    )

    if volume_ratio >= 1.5:

        if candle_direction:

            buy_score += 15

        else:

            sell_score += 15

    elif volume_ratio >= 1.15:

        if candle_direction:

            buy_score += 8

        else:

            sell_score += 8

    # ========================================================
    # PRICE ACTION = 15
    # ========================================================

    recent_high = (
        df["high"]
        .iloc[-21:-1]
        .max()
    )

    recent_low = (
        df["low"]
        .iloc[-21:-1]
        .min()
    )

    current_high = float(
        current["high"]
    )

    current_low = float(
        current["low"]
    )

    current_open = float(
        current["open"]
    )

    current_close = float(
        current["close"]
    )

    # Bullish breakout
    if current_close > recent_high:

        buy_score += 15

    # Bearish breakout
    elif current_close < recent_low:

        sell_score += 15

    else:

        candle_range = (
            current_high -
            current_low
        )

        if candle_range > 0:

            body = abs(
                current_close -
                current_open
            )

            body_ratio = (
                body /
                candle_range
            )

            if body_ratio >= 0.65:

                if current_close > current_open:

                    buy_score += 8

                else:

                    sell_score += 8

    # ========================================================
    # DETERMINE DIRECTION
    # ========================================================

    if buy_score > sell_score:

        direction = "BUY"
        strength = buy_score

    elif sell_score > buy_score:

        direction = "SELL"
        strength = sell_score

    else:

        return None

    # --------------------------------------------------------
    # Minimum strength
    # --------------------------------------------------------

    if strength < MIN_SIGNAL_STRENGTH:

        return None

    # ========================================================
    # MACD MOMENTUM FILTER
    # ========================================================

    if direction == "BUY":

        if (
            macd_hist < previous_hist
            and strength < 80
        ):

            return None

    if direction == "SELL":

        if (
            macd_hist > previous_hist
            and strength < 80
        ):

            return None

    # ========================================================
    # ENTRY
    # ========================================================

    entry = price

    # ========================================================
    # SL / TP
    #
    # Risk = 1 ATR
    # TP1 = 1R
    # TP2 = 2R
    # ========================================================

    risk = atr

    if direction == "BUY":

        sl = entry - risk

        tp1 = entry + risk

        tp2 = entry + (
            risk * 2
        )

    else:

        sl = entry + risk

        tp1 = entry - risk

        tp2 = entry - (
            risk * 2
        )

    return {
        "symbol": symbol,
        "direction": direction,
        "strength": int(strength),
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2
    }


# ============================================================
# PRICE FORMAT
# ============================================================

def format_price(price):

    if price >= 1000:

        return f"{price:,.0f}"

    elif price >= 100:

        return f"{price:,.2f}"

    elif price >= 1:

        return f"{price:,.3f}"

    elif price >= 0.01:

        return f"{price:,.5f}"

    else:

        return f"{price:,.8f}"


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def build_message(signal):

    return (
        f"🔥 SIGNAL — {signal['symbol']}\n\n"
        f"📊 الاتجاه: {signal['direction']}\n"
        f"💪 قوة الإشارة: {signal['strength']}%\n\n"
        f"🎯 Entry: {format_price(signal['entry'])}\n"
        f"🛑 SL: {format_price(signal['sl'])}\n"
        f"🎯 TP1: {format_price(signal['tp1'])}\n"
        f"🎯 TP2: {format_price(signal['tp2'])}"
    )


# ============================================================
# MARKET SCANNER
# ============================================================

def scan_market():

    logging.info(
        "Getting Bybit Linear USDT symbols..."
    )

    symbols = get_symbols()

    if not symbols:

        logging.error(
            "Could not get Bybit symbols"
        )

        return

    logging.info(
        f"Found {len(symbols)} USDT perpetual contracts"
    )

    # --------------------------------------------------------
    # Get 24h turnover
    # --------------------------------------------------------

    tickers = get_tickers()

    # --------------------------------------------------------
    # Rank symbols by liquidity
    # --------------------------------------------------------

    symbols = sorted(
        symbols,
        key=lambda symbol:
            tickers.get(symbol, 0),
        reverse=True
    )

    symbols = symbols[
        :MAX_SYMBOLS
    ]

    logging.info(
        f"Scanning top {len(symbols)} liquid symbols..."
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    for symbol in symbols:

        try:

            signal = analyze_symbol(
                symbol
            )

            if signal is None:
                continue

            # ------------------------------------------------
            # Duplicate protection
            # ------------------------------------------------

            signal_key = (
                signal["direction"],
                signal["strength"],
                round(
                    signal["entry"],
                    8
                )
            )

            if (
                last_signals.get(symbol)
                == signal_key
            ):

                logging.info(
                    f"{symbol}: duplicate ignored"
                )

                continue

            last_signals[
                symbol
            ] = signal_key

            # ------------------------------------------------
            # Build message
            # ------------------------------------------------

            message = build_message(
                signal
            )

            logging.info(
                f"{symbol} -> "
                f"{signal['direction']} "
                f"{signal['strength']}%"
            )

            send_telegram(
                message
            )

        except Exception as e:

            logging.exception(
                f"{symbol} analysis error: {e}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    logging.info(
        "🚀 Futures Signal Bot started"
    )

    scan_market()

    logging.info(
        "✅ Scan finished"
    )
