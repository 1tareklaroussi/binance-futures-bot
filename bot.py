import os
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

TIMEFRAME = "5m"
PERIOD = "5d"

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


SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
    "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD",
    "LINK-USD", "DOT-USD", "MATIC-USD", "LTC-USD",
    "SHIB-USD", "TRX-USD", "BCH-USD", "NEAR-USD",
    "UNI-USD", "ATOM-USD", "XLM-USD", "XMR-USD",
    "ETC-USD", "ICP-USD", "FIL-USD", "HBAR-USD",
    "VET-USD", "APT-USD", "OP-USD", "ARB-USD",
    "INJ-USD", "RNDR-USD", "FTM-USD", "SUI-USD",
    "SEI-USD", "GALA-USD", "SAND-USD", "MANA-USD",
    "AAVE-USD", "SNX-USD", "MKR-USD", "AXS-USD",
    "TIA-USD", "TAO-USD", "KAS-USD", "STX-USD",
    "IMX-USD", "PEPE-USD", "WIF-USD", "BONK-USD",
    "FLOKI-USD", "JUP-USD", "PYTH-USD", "RUNE-USD",
    "ALGO-USD", "EGLD-USD", "QNT-USD", "EOS-USD",
    "XTZ-USD", "FLOW-USD", "THETA-USD", "CRV-USD",
    "LDO-USD", "COMP-USD", "ZEC-USD", "DASH-USD"
]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ============================================================
# FILES
# ============================================================

def ensure_files():

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "w") as f:
            json.dump([], f)

    if not os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "w") as f:
            json.dump({
                "volume_multiplier": BASE_VOLUME_MULTIPLIER,
                "approved": False,
                "last_update": None
            }, f, indent=2)


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

        r = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            },
            timeout=20
        )

        r.raise_for_status()

        logging.info("Telegram message sent")

    except Exception as e:

        logging.error(
            f"Telegram error: {e}"
        )


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
        100 / (1 + rs)
    ).fillna(50)


def atr(df, period=14):

    previous = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous).abs(),
            (df["Low"] - previous).abs()
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

    typical = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    dates = df.index.date

    pv = (
        typical *
        df["Volume"]
    )

    cumulative_pv = (
        pv.groupby(dates)
        .cumsum()
    )

    cumulative_volume = (
        df["Volume"]
        .groupby(dates)
        .cumsum()
    )

    df["VWAP"] = (
        cumulative_pv /
        cumulative_volume
    )

    return df


# ============================================================
# DATA
# ============================================================

def get_data(symbol):

    try:

        df = yf.download(
            symbol,
            period=PERIOD,
            interval=TIMEFRAME,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if df is None or df.empty:
            return None

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = [
                c[0]
                for c in df.columns
            ]

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(
            x in df.columns
            for x in required
        ):

            return None

        df = df[required].copy()

        for c in required:

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

        df.dropna(
            inplace=True
        )

        df.sort_index(
            inplace=True
        )

        # Ignore currently forming candle
        if len(df) > 2:
            df = df.iloc[:-1]

        if len(df) < 100:
            return None

        # Indicators

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

    except Exception as e:

        logging.error(
            f"{symbol}: {e}"
        )

        return None


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
# SIGNAL
# ============================================================

def signal(df):

    if len(df) < 60:
        return None

    c = df.iloc[-1]
    p = df.iloc[-2]

    volume_multiplier = (
        adaptive_volume()
    )

    volume_ok = (
        c["Volume"] >
        c["VOL_MA"] *
        volume_multiplier
    )

    bullish_cross = (
        p["EMA9"] <= p["EMA21"]
        and
        c["EMA9"] > c["EMA21"]
    )

    bearish_cross = (
        p["EMA9"] >= p["EMA21"]
        and
        c["EMA9"] < c["EMA21"]
    )

    buy = (
        bullish_cross
        and
        c["Close"] > c["VWAP"]
        and
        c["Close"] > c["EMA50"]
        and
        52 <= c["RSI"] <= 70
        and
        volume_ok
    )

    sell = (
        bearish_cross
        and
        c["Close"] < c["VWAP"]
        and
        c["Close"] < c["EMA50"]
        and
        30 <= c["RSI"] <= 48
        and
        volume_ok
    )

    if not buy and not sell:
        return None

    direction = (
        "BUY"
        if buy
        else
        "SELL"
    )

    price = float(
        c["Close"]
    )

    current_atr = float(
        c["ATR"]
    )

    if current_atr <= 0:
        return None

    risk = (
        current_atr *
        SL_ATR
    )

    if direction == "BUY":

        sl = price - risk

        tp = (
            price +
            risk * TP_R
        )

    else:

        sl = price + risk

        tp = (
            price -
            risk * TP_R
        )

    return {
        "direction": direction,
        "entry": price,
        "sl": sl,
        "tp": tp,
        "rsi": float(c["RSI"]),
        "volume_ratio": float(
            c["Volume"] /
            c["VOL_MA"]
        ),
        "atr": current_atr,
        "time": str(
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

    for t in trades:

        if (
            t.get("symbol")
            == symbol
            and
            t.get("signal_time")
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

    return trade


# ============================================================
# CHECK OLD OPEN TRADES
# ============================================================

def update_open_trades(
    trades
):

    changed = False

    for trade in trades:

        if trade.get(
            "status"
        ) != "OPEN":

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

        risk = abs(
            float(
                trade["entry"]
            ) - sl
        )

        for _, row in future.iterrows():

            high = float(
                row["High"]
            )

            low = float(
                row["Low"]
            )

            if direction == "BUY":

                # Conservative:
                # SL first if both hit
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
        t for t in trades
        if t.get("status")
        == "CLOSED"
    ]

    if len(closed) < MIN_LEARNING_TRADES:

        return None

    wins = [
        t for t in closed
        if t.get("result")
        == "WIN"
    ]

    losses = [
        t for t in closed
        if t.get("result")
        == "LOSS"
    ]

    if not losses:
        return None

    gross_profit = sum(
        max(
            0,
            float(t["R"])
        )
        for t in closed
    )

    gross_loss = abs(
        sum(
            min(
                0,
                float(t["R"])
            )
            for t in closed
        )
    )

    if gross_loss <= 0:
        return None

    pf = (
        gross_profit /
        gross_loss
    )

    win_rate = (
        len(wins) /
        len(closed)
        * 100
    )

    # --------------------------------------------------------
    # Analyze volume
    # --------------------------------------------------------

    high_volume = [
        t for t in closed
        if float(
            t.get(
                "volume_ratio",
                0
            )
        ) >= 1.35
    ]

    low_volume = [
        t for t in closed
        if float(
            t.get(
                "volume_ratio",
                0
            )
        ) < 1.35
    ]

    def wr(group):

        if not group:
            return 0

        return (
            sum(
                t["result"]
                == "WIN"
                for t in group
            )
            /
            len(group)
            * 100
        )

    high_wr = wr(
        high_volume
    )

    low_wr = wr(
        low_volume
    )

    model = get_model()

    old_multiplier = float(
        model.get(
            "volume_multiplier",
            BASE_VOLUME_MULTIPLIER
        )
    )

    # --------------------------------------------------------
    # Adaptive rule
    #
    # Only consider changing after
    # enough historical trades.
    # --------------------------------------------------------

    if (
        len(high_volume) >= 30
        and
        high_wr >
        low_wr + 5
    ):

        proposed = 1.35

    else:

        proposed = old_multiplier

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

        "pf":
            round(
                pf,
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
# REPORT
# ============================================================

def make_report(
    signals,
    learning
):

    message = (
        "⚡ <b>ADAPTIVE SCALPING BOT</b>\n\n"
        "📡 Yahoo Finance\n"
        "⏱️ 5M\n"
        "🧠 EMA9/21 + VWAP + RSI + Volume\n\n"
    )

    if signals:

        message += (
            "━━━━━━━━━━━━━━━━━━\n"
            "🚨 <b>NEW SIGNALS</b>\n"
        )

        for symbol, sig in signals:

            icon = (
                "🟢"
                if sig["direction"]
                == "BUY"
                else "🔴"
            )

            message += (
                f"\n{icon} <b>{symbol}</b>\n"
                f"Direction: <b>{sig['direction']}</b>\n"
                f"Entry: {sig['entry']:.8g}\n"
                f"SL: {sig['sl']:.8g}\n"
                f"TP: {sig['tp']:.8g}\n"
                f"RSI: {sig['rsi']:.2f}\n"
                f"Volume: {sig['volume_ratio']:.2f}x\n"
                f"RR: 1:{TP_R:.2f}\n"
            )

    if learning:

        message += (
            "\n━━━━━━━━━━━━━━━━━━\n"
            "🧠 <b>LEARNING</b>\n\n"

            f"Trades: {learning['trades']}\n"
            f"Wins: {learning['wins']}\n"
            f"Losses: {learning['losses']}\n"
            f"Win Rate: {learning['win_rate']}%\n"
            f"Profit Factor: {learning['pf']}\n\n"

            f"Volume ≥ 1.35x: "
            f"{learning['high_volume_wr']}% WR\n"

            f"Volume < 1.35x: "
            f"{learning['low_volume_wr']}% WR\n\n"
        )

        if learning["changed"]:

            message += (
                "🟢 <b>ADAPTIVE UPDATE</b>\n"
                f"Volume filter: "
                f"{learning['old_multiplier']}x"
                " → "
                f"{learning['new_multiplier']}x\n"
            )

        else:

            message += (
                "⚪ No parameter update.\n"
            )

    if not signals and not learning:

        message += (
            "⚪ No new signals.\n"
        )

    return message


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_files()

    logging.info(
        "🚀 Adaptive Scalping Bot started"
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
    # 2. Learn
    # --------------------------------------------------------

    learning = learn(
        trades
    )

    # --------------------------------------------------------
    # 3. Scan
    # --------------------------------------------------------

    signals = []

    for symbol in SYMBOLS:

        df = get_data(
            symbol
        )

        if df is None:
            continue

        sig = signal(
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

    # --------------------------------------------------------
    # 4. Save
    # --------------------------------------------------------

    save_json(
        TRADES_FILE,
        trades
    )

    # --------------------------------------------------------
    # 5. Telegram
    # --------------------------------------------------------

    message = make_report(
        signals,
        learning
    )

    telegram(
        message
    )

    logging.info(
        "✅ Scan completed"
    )


if __name__ == "__main__":
    main()
