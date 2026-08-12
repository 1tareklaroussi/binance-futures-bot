import os
import json
import time
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not TWELVE_DATA_API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY is missing")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

if not TELEGRAM_CHAT_ID:
    raise RuntimeError("TELEGRAM_CHAT_ID is missing")


BASE_URL = "https://api.twelvedata.com/time_series"

TIMEFRAME = "5min"

# Number of candles requested from Twelve Data.
# 500 candles = enough for EMA/RSI/ATR calculations.
OUTPUT_SIZE = 500

EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50

RSI_PERIOD = 14
ATR_PERIOD = 14

BASE_VOLUME_MULTIPLIER = 1.20

SL_ATR = 1.20
TP_R = 1.80

MIN_LEARNING_TRADES = 50

TRADES_FILE = "data/trades.json"
MODEL_FILE = "data/model.json"


# ============================================================
# SYMBOLS
# ============================================================

# Twelve Data crypto symbols use BTC/USD format.
# We keep the same coins but convert automatically.

RAW_SYMBOLS = [
    "BTC", "ETH", "SOL", "BNB", "XRP",
    "DOGE", "ADA", "AVAX", "LINK", "DOT",
    "MATIC", "LTC", "SHIB", "TRX", "BCH",
    "NEAR", "UNI", "ATOM", "XLM", "XMR",
    "ETC", "ICP", "FIL", "HBAR", "VET",
    "APT", "OP", "ARB", "INJ", "RNDR",
    "FTM", "SUI", "SEI", "GALA", "SAND",
    "MANA", "AAVE", "SNX", "MKR", "AXS",
    "TIA", "TAO", "KAS", "STX", "IMX",
    "PEPE", "WIF", "BONK", "FLOKI", "JUP",
    "PYTH", "RUNE", "ALGO", "EGLD", "QNT",
    "EOS", "XTZ", "FLOW", "THETA", "CRV",
    "LDO", "COMP", "ZEC", "DASH"
]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ============================================================
# FILE MANAGEMENT
# ============================================================

def ensure_files():

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(TRADES_FILE):

        with open(TRADES_FILE, "w") as f:
            json.dump([], f)

    if not os.path.exists(MODEL_FILE):

        with open(MODEL_FILE, "w") as f:

            json.dump(
                {
                    "volume_multiplier": BASE_VOLUME_MULTIPLIER,
                    "approved": False,
                    "last_update": None
                },
                f,
                indent=2
            )


def load_json(path, default):

    try:

        with open(path, "r") as f:
            return json.load(f)

    except Exception:

        return default


def save_json(path, data):

    with open(path, "w") as f:

        json.dump(
            data,
            f,
            indent=2
        )


# ============================================================
# TELEGRAM
# ============================================================

def telegram(message):

    url = (
        f"https://api.telegram.org/"
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

        logging.info("Telegram message sent")

    except Exception as e:

        logging.error(
            f"Telegram error: {e}"
        )


# ============================================================
# TWELVE DATA
# ============================================================

def get_data(symbol):

    params = {
        "symbol": f"{symbol}/USD",
        "interval": TIMEFRAME,
        "outputsize": OUTPUT_SIZE,
        "apikey": TWELVE_DATA_API_KEY
    }

    for attempt in range(3):

        try:

            response = requests.get(
                BASE_URL,
                params=params,
                timeout=20
            )

            data = response.json()

            if response.status_code == 429:

                logging.warning(
                    f"{symbol}: Twelve Data rate limit. "
                    f"Waiting..."
                )

                time.sleep(15)
                continue

            if data.get("status") == "error":

                logging.warning(
                    f"{symbol}: "
                    f"{data.get('message', 'Unknown error')}"
                )

                return None

            values = data.get("values")

            if not values:

                logging.warning(
                    f"{symbol}: no data returned"
                )

                return None

            df = pd.DataFrame(values)

            required = [
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]

            if not all(
                column in df.columns
                for column in required
            ):

                logging.warning(
                    f"{symbol}: missing OHLCV columns"
                )

                return None

            df["datetime"] = pd.to_datetime(
                df["datetime"],
                errors="coerce"
            )

            df = df.set_index(
                "datetime"
            )

            rename = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }

            df = df.rename(
                columns=rename
            )

            for column in [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

            df.dropna(
                inplace=True
            )

            df.sort_index(
                inplace=True
            )

            # Remove currently forming candle.
            if len(df) > 2:
                df = df.iloc[:-1]

            if len(df) < 100:

                logging.warning(
                    f"{symbol}: insufficient candles"
                )

                return None

            return df

        except Exception as e:

            logging.warning(
                f"{symbol}: request error "
                f"(attempt {attempt + 1}/3): {e}"
            )

            time.sleep(5)

    return None


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


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

    return (
        100 -
        (100 / (1 + rs))
    ).fillna(50)


def atr(df, period=14):

    previous_close = (
        df["Close"].shift(1)
    )

    tr = pd.concat(
        [
            df["High"] - df["Low"],

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
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# VWAP
# ============================================================

def add_vwap(df):

    df = df.copy()

    typical_price = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    session = df.index.date

    pv = (
        typical_price *
        df["Volume"]
    )

    cumulative_pv = (
        pv.groupby(session)
        .cumsum()
    )

    cumulative_volume = (
        df["Volume"]
        .groupby(session)
        .cumsum()
    )

    df["VWAP"] = (
        cumulative_pv /
        cumulative_volume
    )

    return df


# ============================================================
# BUILD INDICATORS
# ============================================================

def prepare_data(df):

    df = df.copy()

    df["EMA9"] = ema(
        df["Close"],
        EMA_FAST
    )

    df["EMA21"] = ema(
        df["Close"],
        EMA_SLOW
    )

    df["EMA50"] = ema(
        df["Close"],
        EMA_TREND
    )

    df["RSI"] = rsi(
        df["Close"],
        RSI_PERIOD
    )

    df["ATR"] = atr(
        df,
        ATR_PERIOD
    )

    df["VOL_MA"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df = add_vwap(df)

    return df


# ============================================================
# ADAPTIVE MODEL
# ============================================================

def get_model():

    return load_json(
        MODEL_FILE,
        {
            "volume_multiplier":
                BASE_VOLUME_MULTIPLIER,

            "approved":
                False,

            "last_update":
                None
        }
    )


def adaptive_volume():

    model = get_model()

    value = float(
        model.get(
            "volume_multiplier",
            BASE_VOLUME_MULTIPLIER
        )
    )

    return max(
        1.0,
        min(
            value,
            2.0
        )
    )


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(df):

    if len(df) < 60:
        return None

    df = prepare_data(df)

    current = df.iloc[-1]
    previous = df.iloc[-2]

    volume_multiplier = (
        adaptive_volume()
    )

    volume_ratio = (
        current["Volume"] /
        current["VOL_MA"]
    )

    volume_confirmed = (
        volume_ratio >=
        volume_multiplier
    )

    bullish_cross = (
        previous["EMA9"]
        <= previous["EMA21"]
        and
        current["EMA9"]
        > current["EMA21"]
    )

    bearish_cross = (
        previous["EMA9"]
        >= previous["EMA21"]
        and
        current["EMA9"]
        < current["EMA21"]
    )

    buy = (
        bullish_cross
        and
        current["Close"]
        > current["VWAP"]
        and
        current["Close"]
        > current["EMA50"]
        and
        52 <= current["RSI"] <= 70
        and
        volume_confirmed
    )

    sell = (
        bearish_cross
        and
        current["Close"]
        < current["VWAP"]
        and
        current["Close"]
        < current["EMA50"]
        and
        30 <= current["RSI"] <= 48
        and
        volume_confirmed
    )

    if not buy and not sell:
        return None

    direction = (
        "BUY"
        if buy
        else
        "SELL"
    )

    entry = float(
        current["Close"]
    )

    current_atr = float(
        current["ATR"]
    )

    if current_atr <= 0:
        return None

    risk = (
        current_atr *
        SL_ATR
    )

    if direction == "BUY":

        sl = entry - risk

        tp = (
            entry +
            risk * TP_R
        )

    else:

        sl = entry + risk

        tp = (
            entry -
            risk * TP_R
        )

    return {

        "direction":
            direction,

        "entry":
            entry,

        "sl":
            sl,

        "tp":
            tp,

        "rsi":
            float(
                current["RSI"]
            ),

        "volume_ratio":
            float(
                volume_ratio
            ),

        "atr":
            current_atr,

        "time":
            str(
                df.index[-1]
            )
    }


# ============================================================
# DUPLICATE PROTECTION
# ============================================================

def already_signaled(
    trades,
    symbol,
    signal_time
):

    for trade in trades:

        if (
            trade.get("symbol")
            == symbol
            and
            trade.get("signal_time")
            == signal_time
        ):

            return True

    return False


# ============================================================
# RECORD SIGNAL
# ============================================================

def record_signal(
    trades,
    symbol,
    sig
):

    trade = {

        "id":
            f"{symbol}_{sig['time']}",

        "symbol":
            symbol,

        "signal_time":
            sig["time"],

        "direction":
            sig["direction"],

        "entry":
            sig["entry"],

        "sl":
            sig["sl"],

        "tp":
            sig["tp"],

        "rsi":
            sig["rsi"],

        "volume_ratio":
            sig["volume_ratio"],

        "atr":
            sig["atr"],

        "status":
            "OPEN",

        "result":
            None,

        "R":
            None
    }

    trades.append(
        trade
    )


# ============================================================
# UPDATE OPEN TRADES
# ============================================================

def update_open_trades(trades):

    changed = False

    for trade in trades:

        if (
            trade.get("status")
            != "OPEN"
        ):
            continue

        df = get_data(
            trade["symbol"]
        )

        if df is None:
            continue

        entry_time = pd.to_datetime(
            trade["signal_time"]
        )

        future = df[
            df.index >
            entry_time
        ]

        if future.empty:
            continue

        direction = (
            trade["direction"]
        )

        sl = float(
            trade["sl"]
        )

        tp = float(
            trade["tp"]
        )

        for _, candle in future.iterrows():

            high = float(
                candle["High"]
            )

            low = float(
                candle["Low"]
            )

            if direction == "BUY":

                if low <= sl:

                    trade["status"] = "CLOSED"
                    trade["result"] = "LOSS"
                    trade["R"] = -1.0

                    changed = True
                    break

                if high >= tp:

                    trade["status"] = "CLOSED"
                    trade["result"] = "WIN"
                    trade["R"] = TP_R

                    changed = True
                    break

            else:

                if high >= sl:

                    trade["status"] = "CLOSED"
                    trade["result"] = "LOSS"
                    trade["R"] = -1.0

                    changed = True
                    break

                if low <= tp:

                    trade["status"] = "CLOSED"
                    trade["result"] = "WIN"
                    trade["R"] = TP_R

                    changed = True
                    break

    return changed


# ============================================================
# LEARNING
# ============================================================

def learn(trades):

    closed = [
        trade
        for trade in trades
        if trade.get("status")
        == "CLOSED"
    ]

    if len(closed) < MIN_LEARNING_TRADES:

        return None

    wins = [
        trade
        for trade in closed
        if trade.get("result")
        == "WIN"
    ]

    losses = [
        trade
        for trade in closed
        if trade.get("result")
        == "LOSS"
    ]

    if not losses:
        return None

    gross_profit = sum(
        max(
            0,
            float(trade["R"])
        )
        for trade in closed
    )

    gross_loss = abs(
        sum(
            min(
                0,
                float(trade["R"])
            )
            for trade in closed
        )
    )

    if gross_loss <= 0:
        return None

    profit_factor = (
        gross_profit /
        gross_loss
    )

    win_rate = (
        len(wins) /
        len(closed)
        * 100
    )

    high_volume = [
        trade
        for trade in closed
        if float(
            trade.get(
                "volume_ratio",
                0
            )
        ) >= 1.35
    ]

    low_volume = [
        trade
        for trade in closed
        if float(
            trade.get(
                "volume_ratio",
                0
            )
        ) < 1.35
    ]

    def group_winrate(group):

        if not group:
            return 0

        return (
            sum(
                x["result"]
                == "WIN"
                for x in group
            )
            /
            len(group)
            * 100
        )

    high_wr = group_winrate(
        high_volume
    )

    low_wr = group_winrate(
        low_volume
    )

    model = get_model()

    old_multiplier = float(
        model.get(
            "volume_multiplier",
            BASE_VOLUME_MULTIPLIER
        )
    )

    proposed = old_multiplier

    if (
        len(high_volume) >= 30
        and
        high_wr >
        low_wr + 5
    ):

        proposed = 1.35

    changed = (
        proposed !=
        old_multiplier
    )

    if changed:

        model[
            "volume_multiplier"
        ] = proposed

        model[
            "approved"
        ] = True

        model[
            "last_update"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

        save_json(
            MODEL_FILE,
            model
        )

    return {

        "trades":
            len(closed),

        "wins":
            len(wins),

        "losses":
            len(losses),

        "win_rate":
            round(
                win_rate,
                2
            ),

        "profit_factor":
            round(
                profit_factor,
                3
            ),

        "high_volume_wr":
            round(
                high_wr,
                2
            ),

        "low_volume_wr":
            round(
                low_wr,
                2
            ),

        "old_multiplier":
            old_multiplier,

        "new_multiplier":
            proposed,

        "changed":
            changed
    }


# ============================================================
# TELEGRAM REPORT
# ============================================================

def build_report(
    signals,
    learning
):

    message = (
        "⚡ <b>ADAPTIVE SCALPING V3</b>\n\n"
        "📡 Twelve Data\n"
        "⏱️ 5M\n"
        "🧠 EMA9/21 + VWAP + RSI + Volume\n"
    )

    if signals:

        message += (
            "\n━━━━━━━━━━━━━━━━━━\n"
            "🚨 <b>NEW SIGNAL</b>\n"
        )

        for symbol, sig in signals:

            icon = (
                "🟢"
                if sig["direction"]
                == "BUY"
                else "🔴"
            )

            message += (

                f"\n{icon} "
                f"<b>{symbol}/USD</b>\n"

                f"Direction: "
                f"<b>{sig['direction']}</b>\n"

                f"Entry: "
                f"{sig['entry']:.8g}\n"

                f"SL: "
                f"{sig['sl']:.8g}\n"

                f"TP: "
                f"{sig['tp']:.8g}\n"

                f"RSI: "
                f"{sig['rsi']:.2f}\n"

                f"Volume: "
                f"{sig['volume_ratio']:.2f}x\n"

                f"RR: 1:{TP_R:.2f}\n"
            )

    else:

        message += (
            "\n⚪ No new signals."
        )

    if learning:

        message += (

            "\n\n━━━━━━━━━━━━━━━━━━\n"

            "🧠 <b>LEARNING REPORT</b>\n\n"

            f"Trades: "
            f"{learning['trades']}\n"

            f"Wins: "
            f"{learning['wins']}\n"

            f"Losses: "
            f"{learning['losses']}\n"

            f"Win Rate: "
            f"{learning['win_rate']}%\n"

            f"Profit Factor: "
            f"{learning['profit_factor']}\n\n"

            f"Volume ≥ 1.35x: "
            f"{learning['high_volume_wr']}% WR\n"

            f"Volume < 1.35x: "
            f"{learning['low_volume_wr']}% WR\n"
        )

        if learning["changed"]:

            message += (

                "\n🟢 <b>ADAPTIVE UPDATE</b>\n"

                f"Volume filter: "
                f"{learning['old_multiplier']}x"
                " → "
                f"{learning['new_multiplier']}x"
            )

        else:

            message += (
                "\n⚪ No parameter update."
            )

    return message


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_files()

    logging.info(
        "🚀 Adaptive Scalping V3 started"
    )

    logging.info(
        "📡 Data source: Twelve Data"
    )

    logging.info(
        f"⏱️ Timeframe: {TIMEFRAME}"
    )

    trades = load_json(
        TRADES_FILE,
        []
    )

    # --------------------------------------------------------
    # 1. Update previous trades
    # --------------------------------------------------------

    changed = update_open_trades(
        trades
    )

    if changed:

        save_json(
            TRADES_FILE,
            trades
        )

    # --------------------------------------------------------
    # 2. Learning
    # --------------------------------------------------------

    learning = learn(
        trades
    )

    # --------------------------------------------------------
    # 3. Scan
    # --------------------------------------------------------

    signals = []

    for symbol in RAW_SYMBOLS:

        logging.info(
            f"🔍 Checking {symbol}/USD"
        )

        df = get_data(
            symbol
        )

        if df is None:
            continue

        sig = generate_signal(
            df
        )

        if sig is None:
            continue

        if already_signaled(
            trades,
            symbol,
            sig["time"]
        ):

            continue

        record_signal(
            trades,
            symbol,
            sig
        )

        signals.append(
            (
                symbol,
                sig
            )
        )

        # Small delay to be gentle
        # with API limits.
        time.sleep(0.15)

    # --------------------------------------------------------
    # 4. Save memory
    # --------------------------------------------------------

    save_json(
        TRADES_FILE,
        trades
    )

    # --------------------------------------------------------
    # 5. Telegram
    # --------------------------------------------------------

    telegram(
        build_report(
            signals,
            learning
        )
    )

    logging.info(
        "✅ Scan completed"
    )


if __name__ == "__main__":
    main()
