import os
import time
import hmac
import hashlib
import logging
import requests
import pandas as pd
import numpy as np
from urllib.parse import urlencode


# ============================================================
# CONFIG
# ============================================================

BINANCE_BASE = "https://fapi.binance.com"

BINANCE_API_KEY = os.environ["BINANCE_API_KEY"]
BINANCE_API_SECRET = os.environ["BINANCE_API_SECRET"]

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

INTERVAL = "1h"
KLINE_LIMIT = 250

MIN_SIGNAL_STRENGTH = 70
MAX_SYMBOLS = 40

REQUEST_DELAY = 0.15
MAX_RETRIES = 3


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

session = requests.Session()

session.headers.update({
    "X-MBX-APIKEY": BINANCE_API_KEY,
    "User-Agent": "CryptoSignalBot/1.0"
})


# ============================================================
# BINANCE REQUEST
# ============================================================

def binance_get(path, params=None, signed=False):

    params = params.copy() if params else {}

    if signed:

        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000

        query = urlencode(params)

        signature = hmac.new(
            BINANCE_API_SECRET.encode(),
            query.encode(),
            hashlib.sha256
        ).hexdigest()

        params["signature"] = signature

    for attempt in range(MAX_RETRIES):

        try:

            response = session.get(
                BINANCE_BASE + path,
                params=params,
                timeout=15
            )

            # ------------------------------------------------
            # Geographic / access restriction
            # ------------------------------------------------

            if response.status_code in (403, 451):

                logging.error(
                    "Binance rejected the GitHub Actions connection: "
                    f"HTTP {response.status_code}"
                )

                logging.error(
                    "This is an access/geographic restriction, "
                    "not a Python error."
                )

                return None

            # ------------------------------------------------
            # Rate limit
            # ------------------------------------------------

            if response.status_code == 429:

                wait = min(
                    2 ** attempt,
                    30
                )

                logging.warning(
                    f"Binance 429 - retrying in {wait}s"
                )

                time.sleep(wait)
                continue

            # ------------------------------------------------
            # Server errors
            # ------------------------------------------------

            if response.status_code >= 500:

                wait = min(
                    2 ** attempt,
                    30
                )

                logging.warning(
                    f"Binance server error "
                    f"{response.status_code} - "
                    f"retrying in {wait}s"
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

            time.sleep(REQUEST_DELAY)

            return response.json()

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

    try:

        response = session.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message
            },
            timeout=15
        )

        response.raise_for_status()

        logging.info(
            "Telegram message sent."
        )

        return True

    except requests.RequestException as e:

        logging.error(
            f"Telegram error: {e}"
        )

        return False


# ============================================================
# TEST BINANCE CONNECTION
# ============================================================

def test_binance():

    logging.info(
        "Testing authenticated Binance Futures connection..."
    )

    data = binance_get(
        "/fapi/v2/account",
        signed=True
    )

    if data is None:

        logging.error(
            "❌ Binance Futures connection failed."
        )

        return False

    logging.info(
        "✅ Binance Futures API connection works."
    )

    return True


# ============================================================
# SYMBOLS
# ============================================================

def get_symbols():

    data = binance_get(
        "/fapi/v1/exchangeInfo"
    )

    if not data:
        return []

    symbols = []

    for item in data.get(
        "symbols",
        []
    ):

        if (
            item.get("status") == "TRADING"
            and item.get("contractType")
            == "PERPETUAL"
            and item.get("quoteAsset")
            == "USDT"
        ):

            symbols.append(
                item["symbol"]
            )

    return symbols


# ============================================================
# 24H VOLUME
# ============================================================

def get_volume_rank():

    data = binance_get(
        "/fapi/v1/ticker/24hr"
    )

    if not data:
        return {}

    result = {}

    for item in data:

        symbol = item.get(
            "symbol"
        )

        try:

            result[symbol] = float(
                item.get(
                    "quoteVolume",
                    0
                )
            )

        except:

            result[symbol] = 0

    return result


# ============================================================
# KLINES
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

    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume"
    ]

    for column in numeric:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna()

    return df


# ============================================================
# EMA
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


# ============================================================
# RSI
# ============================================================

def rsi(series, period=14):

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

    result = 100 - (
        100 / (1 + rs)
    )

    return result.fillna(50)


# ============================================================
# MACD
# ============================================================

def macd(series):

    fast = ema(
        series,
        12
    )

    slow = ema(
        series,
        26
    )

    line = fast - slow

    signal = ema(
        line,
        9
    )

    histogram = (
        line -
        signal
    )

    return line, signal, histogram


# ============================================================
# ATR
# ============================================================

def atr(df, period=14):

    previous_close = (
        df["close"].shift(1)
    )

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"] -
                previous_close
            ).abs(),
            (
                df["low"] -
                previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# ANALYSIS
# ============================================================

def analyze(symbol):

    df = get_klines(symbol)

    if df is None or len(df) < 220:
        return None

    # Remove currently open candle
    df = df.iloc[:-1].copy()

    close = df["close"]

    df["ema50"] = ema(
        close,
        50
    )

    df["ema200"] = ema(
        close,
        200
    )

    (
        df["macd"],
        df["macd_signal"],
        df["hist"]
    ) = macd(close)

    df["rsi"] = rsi(
        close
    )

    df["atr"] = atr(
        df
    )

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

    macd_value = float(
        current["macd"]
    )

    macd_signal = float(
        current["macd_signal"]
    )

    hist = float(
        current["hist"]
    )

    previous_hist = float(
        previous["hist"]
    )

    current_rsi = float(
        current["rsi"]
    )

    current_atr = float(
        current["atr"]
    )

    if current_atr <= 0:
        return None

    buy = 0
    sell = 0

    # --------------------------------------------------------
    # EMA = 25
    # --------------------------------------------------------

    if ema50 > ema200:

        buy += 25

    elif ema50 < ema200:

        sell += 25

    # --------------------------------------------------------
    # MACD = 25
    # --------------------------------------------------------

    if (
        macd_value > macd_signal
        and hist > 0
    ):

        buy += 25

    elif (
        macd_value < macd_signal
        and hist < 0
    ):

        sell += 25

    # --------------------------------------------------------
    # RSI = 20
    # --------------------------------------------------------

    if 50 <= current_rsi <= 68:

        buy += 20

    elif 32 <= current_rsi < 50:

        sell += 20

    elif current_rsi > 68:

        buy += 10

    elif current_rsi < 32:

        sell += 10

    # --------------------------------------------------------
    # VOLUME = 15
    # --------------------------------------------------------

    avg_volume = (
        df["volume"]
        .iloc[-21:-1]
        .mean()
    )

    volume_ratio = (
        float(current["volume"]) /
        avg_volume
        if avg_volume > 0
        else 0
    )

    if volume_ratio >= 1.5:

        if current["close"] > current["open"]:

            buy += 15

        else:

            sell += 15

    elif volume_ratio >= 1.15:

        if current["close"] > current["open"]:

            buy += 8

        else:

            sell += 8

    # --------------------------------------------------------
    # PRICE ACTION = 15
    # --------------------------------------------------------

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

    if price > recent_high:

        buy += 15

    elif price < recent_low:

        sell += 15

    else:

        candle_range = (
            current["high"] -
            current["low"]
        )

        if candle_range > 0:

            body = abs(
                current["close"] -
                current["open"]
            )

            body_ratio = (
                body /
                candle_range
            )

            if body_ratio >= 0.65:

                if current["close"] > current["open"]:

                    buy += 8

                else:

                    sell += 8

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if buy > sell:

        direction = "BUY"
        strength = buy

    elif sell > buy:

        direction = "SELL"
        strength = sell

    else:

        return None

    if strength < MIN_SIGNAL_STRENGTH:
        return None

    # --------------------------------------------------------
    # MACD momentum confirmation
    # --------------------------------------------------------

    if direction == "BUY":

        if (
            hist < previous_hist
            and strength < 80
        ):
            return None

    if direction == "SELL":

        if (
            hist > previous_hist
            and strength < 80
        ):
            return None

    # --------------------------------------------------------
    # Entry / SL / TP
    # --------------------------------------------------------

    entry = price

    risk = current_atr

    if direction == "BUY":

        sl = entry - risk
        tp1 = entry + risk
        tp2 = entry + (2 * risk)

    else:

        sl = entry + risk
        tp1 = entry - risk
        tp2 = entry - (2 * risk)

    return {
        "symbol": symbol,
        "direction": direction,
        "strength": strength,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2
    }


# ============================================================
# FORMAT PRICE
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
# TELEGRAM MESSAGE
# ============================================================

def make_message(signal):

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
# MAIN SCAN
# ============================================================

def scan():

    symbols = get_symbols()

    if not symbols:

        logging.error(
            "❌ Could not retrieve Binance Futures symbols."
        )

        return False

    logging.info(
        f"Found {len(symbols)} Futures symbols."
    )

    volumes = get_volume_rank()

    symbols.sort(
        key=lambda s:
        volumes.get(s, 0),
        reverse=True
    )

    symbols = symbols[
        :MAX_SYMBOLS
    ]

    logging.info(
        f"Analyzing top {len(symbols)} symbols..."
    )

    signals_sent = 0

    for symbol in symbols:

        try:

            signal = analyze(
                symbol
            )

            if signal is None:
                continue

            message = make_message(
                signal
            )

            logging.info(
                f"🔥 {symbol} "
                f"{signal['direction']} "
                f"{signal['strength']}%"
            )

            if send_telegram(message):

                signals_sent += 1

        except Exception as e:

            logging.exception(
                f"{symbol}: {e}"
            )

    logging.info(
        f"Scan complete. "
        f"Signals sent: {signals_sent}"
    )

    return True


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    logging.info(
        "🚀 Binance Futures Signal Bot"
    )

    if not test_binance():

        raise SystemExit(
            "Binance connection failed."
        )

    scan()

    logging.info(
        "🏁 Finished."
    )
