import os
import time
import logging
import requests
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

BINANCE_BASE = "https://fapi.binance.com"

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

INTERVAL = "1h"
KLINE_LIMIT = 250

# أقل قوة لإرسال إشارة
MIN_SIGNAL_STRENGTH = 70

# عدد العملات التي نفحصها في كل دورة
MAX_SYMBOLS = 40

# الانتظار بين طلبات Binance
REQUEST_DELAY = 0.15

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

session = requests.Session()

# لتجنب إرسال نفس الإشارة عدة مرات
last_signals = {}


# ============================================================
# BINANCE REQUEST
# ============================================================

def binance_get(path, params=None, retries=5):

    for attempt in range(retries):

        try:
            response = session.get(
                BINANCE_BASE + path,
                params=params,
                timeout=15
            )

            # Rate limit
            if response.status_code == 429:
                wait = min(2 ** attempt, 30)

                logging.warning(
                    f"Binance 429 - waiting {wait}s"
                )

                time.sleep(wait)
                continue

            # Temporary server errors
            if response.status_code in (418, 500, 502, 503, 504):
                wait = min(2 ** attempt, 30)

                logging.warning(
                    f"Binance HTTP {response.status_code} - waiting {wait}s"
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

            time.sleep(REQUEST_DELAY)

            return response.json()

        except requests.RequestException as e:

            wait = min(2 ** attempt, 30)

            logging.warning(
                f"Request error: {e} | retry in {wait}s"
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

        r = session.post(
            url,
            data=payload,
            timeout=15
        )

        r.raise_for_status()

        logging.info("Telegram signal sent")

        return True

    except requests.RequestException as e:

        logging.error(
            f"Telegram error: {e}"
        )

        return False


# ============================================================
# GET FUTURES SYMBOLS
# ============================================================

def get_symbols():

    data = binance_get(
        "/fapi/v1/exchangeInfo"
    )

    if not data:
        return []

    symbols = []

    for item in data["symbols"]:

        if (
            item["status"] == "TRADING"
            and item["contractType"] == "PERPETUAL"
            and item["quoteAsset"] == "USDT"
        ):
            symbols.append(item["symbol"])

    return symbols


# ============================================================
# GET 24H VOLUME
# ============================================================

def get_volume_rank():

    data = binance_get(
        "/fapi/v1/ticker/24hr"
    )

    if not data:
        return {}

    result = {}

    for item in data:

        symbol = item["symbol"]

        if symbol.endswith("USDT"):

            try:
                volume = float(
                    item["quoteVolume"]
                )

                result[symbol] = volume

            except:
                pass

    return result


# ============================================================
# GET KLINES
# ============================================================

def get_klines(symbol):

    data = binance_get(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": KLINE_LIMIT
        }
    )

    if not data:
        return None

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
        "taker_buy_base",
        "taker_buy_quote",
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
        "quote_volume"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


# ============================================================
# EMA
# ============================================================

def calculate_ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, period=14):

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

    rsi = 100 - (
        100 / (1 + rs)
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

    ema_fast = calculate_ema(
        series,
        fast
    )

    ema_slow = calculate_ema(
        series,
        slow
    )

    macd = ema_fast - ema_slow

    signal_line = calculate_ema(
        macd,
        signal
    )

    histogram = macd - signal_line

    return macd, signal_line, histogram


# ============================================================
# ATR
# ============================================================

def calculate_atr(df, period=14):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - previous_close
    ).abs()

    tr3 = (
        low - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
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

    if df is None or len(df) < 220:
        return None

    # --------------------------------------------------------
    # IMPORTANT:
    # Ignore the currently forming candle.
    # --------------------------------------------------------

    df = df.iloc[:-1].copy()

    if len(df) < 200:
        return None

    close = df["close"]

    # Indicators
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
    ) = calculate_macd(close)

    df["rsi"] = calculate_rsi(
        close,
        14
    )

    df["atr"] = calculate_atr(
        df,
        14
    )

    # --------------------------------------------------------
    # Current candle
    # --------------------------------------------------------

    current = df.iloc[-1]
    previous = df.iloc[-2]

    price = float(current["close"])

    ema50 = float(current["ema50"])
    ema200 = float(current["ema200"])

    macd = float(current["macd"])
    macd_signal = float(
        current["macd_signal"]
    )

    macd_hist = float(
        current["macd_hist"]
    )

    previous_hist = float(
        previous["macd_hist"]
    )

    rsi = float(current["rsi"])

    atr = float(current["atr"])

    if atr <= 0:
        return None

    # ========================================================
    # SCORE SYSTEM
    # ========================================================

    buy_score = 0
    sell_score = 0

    # --------------------------------------------------------
    # EMA TREND = 25 points
    # --------------------------------------------------------

    if ema50 > ema200:
        buy_score += 25

    elif ema50 < ema200:
        sell_score += 25

    # --------------------------------------------------------
    # MACD = 25 points
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RSI = 20 points
    # --------------------------------------------------------

    # We don't want to buy when RSI is extremely overbought.
    if 50 <= rsi <= 68:
        buy_score += 20

    elif 32 <= rsi <= 50:
        sell_score += 20

    # Strong momentum continuation
    elif rsi > 68:
        buy_score += 10

    elif rsi < 32:
        sell_score += 10

    # --------------------------------------------------------
    # VOLUME = 15 points
    # --------------------------------------------------------

    avg_volume = df["volume"].iloc[-21:-1].mean()

    current_volume = float(
        current["volume"]
    )

    volume_ratio = (
        current_volume / avg_volume
        if avg_volume > 0
        else 0
    )

    if volume_ratio >= 1.5:

        # Strong volume
        if close.iloc[-1] > close.iloc[-2]:
            buy_score += 15

        elif close.iloc[-1] < close.iloc[-2]:
            sell_score += 15

    elif volume_ratio >= 1.15:

        if close.iloc[-1] > close.iloc[-2]:
            buy_score += 8

        elif close.iloc[-1] < close.iloc[-2]:
            sell_score += 8

    # --------------------------------------------------------
    # PRICE ACTION = 15 points
    # --------------------------------------------------------

    recent_high = df["high"].iloc[-21:-1].max()
    recent_low = df["low"].iloc[-21:-1].min()

    current_high = float(
        current["high"]
    )

    current_low = float(
        current["low"]
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

        # Momentum candle
        candle_range = (
            current_high - current_low
        )

        if candle_range > 0:

            body = abs(
                current_close
                - float(current["open"])
            )

            body_ratio = (
                body / candle_range
            )

            if body_ratio >= 0.65:

                if current_close > float(
                    current["open"]
                ):
                    buy_score += 8

                else:
                    sell_score += 8

    # ========================================================
    # DETERMINE SIGNAL
    # ========================================================

    if buy_score > sell_score:

        direction = "BUY"
        strength = buy_score

    elif sell_score > buy_score:

        direction = "SELL"
        strength = sell_score

    else:
        return None

    # Don't send weak signals
    if strength < MIN_SIGNAL_STRENGTH:
        return None

    # ========================================================
    # MACD MOMENTUM CONFIRMATION
    # ========================================================

    # Avoid some signals where MACD is losing momentum
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
    # STOP LOSS / TAKE PROFIT
    #
    # Risk = 1 ATR
    # TP1 = 1.0R
    # TP2 = 2.0R
    # ========================================================

    risk = atr * 1.0

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

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "symbol": symbol,
        "direction": direction,
        "strength": int(strength),
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rsi": rsi,
        "atr": atr
    }


# ============================================================
# FORMAT PRICE
# ============================================================

def format_price(price):

    if price >= 1000:
        return f"{price:,.0f}"

    elif price >= 100:
        return f"{price:,.2f}"

    elif price >= 1:
        return f"{price:,.3f}"

    else:
        return f"{price:,.6f}"


# ============================================================
# BUILD TELEGRAM MESSAGE
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
# MAIN SCANNER
# ============================================================

def scan_market():

    logging.info("Getting Futures symbols...")

    symbols = get_symbols()

    if not symbols:

        logging.error(
            "Could not get Binance symbols"
        )

        return

    logging.info(
        f"Found {len(symbols)} Futures symbols"
    )

    # --------------------------------------------------------
    # Rank by 24h volume
    # --------------------------------------------------------

    volumes = get_volume_rank()

    symbols = sorted(
        symbols,
        key=lambda x: volumes.get(x, 0),
        reverse=True
    )

    # Only analyze the most liquid symbols
    symbols = symbols[:MAX_SYMBOLS]

    logging.info(
        f"Scanning top {len(symbols)} symbols..."
    )

    for symbol in symbols:

        try:

            signal = analyze_symbol(
                symbol
            )

            if signal is None:
                continue

            # ------------------------------------------------
            # Prevent duplicate signal
            # ------------------------------------------------

            signal_key = (
                signal["direction"],
                signal["strength"]
            )

            if last_signals.get(symbol) == signal_key:

                logging.info(
                    f"{symbol}: duplicate signal ignored"
                )

                continue

            # Save signal
            last_signals[
                symbol
            ] = signal_key

            # ------------------------------------------------
            # Telegram
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
                f"Error analyzing {symbol}: {e}"
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    logging.info(
        "🚀 Futures Signal Bot started"
    )

    while True:

        try:

            scan_market()

        except Exception as e:

            logging.exception(
                f"Main loop error: {e}"
            )

        # ----------------------------------------------------
        # Scan every 10 minutes.
        #
        # The bot uses only CLOSED 1H candles,
        # so scanning more frequently doesn't create
        # new candle signals.
        # ----------------------------------------------------

        logging.info(
            "Waiting 10 minutes..."
        )

        time.sleep(600)
