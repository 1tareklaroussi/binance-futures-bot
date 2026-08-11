# ============================================================
# 🚀 CRYPTO STRATEGY BACKTEST V5
#
# STRATEGY:
# 4H TREND
#     ↓
# 1H BREAKOUT
#     ↓
# 15M RETEST
#     ↓
# RSI + VOLUME + CANDLE
#     ↓
# ENTRY
#
# DATA:
# Binance Futures Historical Klines
#
# OUTPUT:
# - Win Rate
# - Loss Rate
# - Profit Factor
# - Net R
# - Average R
# - Max Drawdown
# - Max Consecutive Losses
# - BUY / SELL statistics
# - CSV results
#
# IMPORTANT:
# This is a backtest simulator.
# It does NOT guarantee future performance.
# ============================================================

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timezone


# ============================================================
# ⚙️ SETTINGS
# ============================================================

BASE_URL = "https://fapi.binance.com"

# How many days to test
BACKTEST_DAYS = 180

# Timeframes
TF_1H = "1h"
TF_15M = "15m"

# Indicators
EMA_FAST = 50
EMA_TREND = 200

RSI_PERIOD = 14
ATR_PERIOD = 14

VOLUME_PERIOD = 20
VOLUME_MULTIPLIER = 1.20

BREAKOUT_LOOKBACK = 20

# Retest
RETEST_ATR_TOLERANCE = 0.35

# SL
ATR_SL_MULTIPLIER = 1.20

# Take profits
TP1_R = 1.5
TP2_R = 2.5
TP3_R = 3.5

# Minimum setup score
MIN_SCORE = 7

# Initial capital
INITIAL_BALANCE = 10000.0

# Risk per trade
RISK_PERCENT = 1.0

# Fees
# Change according to your actual futures account.
FEE_RATE = 0.0004

# Slippage simulation
SLIPPAGE = 0.0002

# Maximum bars to keep a trade open
MAX_TRADE_BARS = 96


# ============================================================
# 🪙 SYMBOLS
# ============================================================

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "SHIBUSDT",
    "TRXUSDT",
    "BCHUSDT",
    "NEARUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "XLMUSDT",
    "XMRUSDT",
    "ETCUSDT",
    "ICPUSDT",
    "FILUSDT",
    "HBARUSDT",
    "VETUSDT",
    "APTUSDT",
    "OPUSDT",
    "ARBUSDT",
    "INJUSDT",
    "RENDERUSDT",
    "SUIUSDT",
    "SEIUSDT",
    "GALAUSDT",
    "SANDUSDT",
    "MANAUSDT",
    "AAVEUSDT",
    "SNXUSDT",
    "MKRUSDT",
    "AXSUSDT",
    "TIAUSDT",
    "TAOUSDT",
    "KASUSDT",
    "STXUSDT",
    "IMXUSDT",
    "PEPEUSDT",
    "WIFUSDT",
    "BONKUSDT",
    "FLOKIUSDT",
    "JUPUSDT",
    "PYTHUSDT",
    "RUNEUSDT",
    "ALGOUSDT",
    "EGLDUSDT",
    "QNTUSDT",
    "EOSUSDT",
    "XTZUSDT",
    "FLOWUSDT",
    "THETAUSDT",
    "CRVUSDT",
    "LDOUSDT",
    "COMPUSDT",
    "ZECUSDT",
    "DASHUSDT"
]


# ============================================================
# 📡 BINANCE DATA
# ============================================================

def get_klines(symbol, interval, start_ms, end_ms):

    all_data = []

    current_start = start_ms

    while current_start < end_ms:

        url = f"{BASE_URL}/fapi/v1/klines"

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000
        }

        try:

            response = requests.get(
                url,
                params=params,
                timeout=20
            )

            response.raise_for_status()

            data = response.json()

        except Exception as e:

            print(
                f"❌ {symbol} {interval} "
                f"download error: {e}"
            )

            time.sleep(3)

            continue

        if not data:
            break

        all_data.extend(data)

        last_open_time = data[-1][0]

        if last_open_time <= current_start:
            break

        current_start = last_open_time + 1

        time.sleep(0.15)

    if not all_data:
        return None

    columns = [
        "OpenTime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "CloseTime",
        "QuoteVolume",
        "Trades",
        "TakerBuyVolume",
        "TakerBuyQuoteVolume",
        "Ignore"
    ]

    df = pd.DataFrame(
        all_data,
        columns=columns
    )

    df["OpenTime"] = pd.to_datetime(
        df["OpenTime"],
        unit="ms",
        utc=True
    )

    df.set_index(
        "OpenTime",
        inplace=True
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df[numeric_columns]

    df.dropna(inplace=True)

    df = df[~df.index.duplicated(
        keep="first"
    )]

    return df


# ============================================================
# 📊 INDICATORS
# ============================================================

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


def ATR(df, period=14):

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
# 🕓 BUILD 4H FROM 1H
# ============================================================

def make_4h(df):

    result = df.resample(
        "4h"
    ).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }
    )

    result.dropna(
        inplace=True
    )

    return result


# ============================================================
# 🕯️ CANDLE CONFIRMATION
# ============================================================

def bullish_candle(candle):

    candle_range = (
        candle["High"] -
        candle["Low"]
    )

    if candle_range <= 0:
        return False

    body = abs(
        candle["Close"] -
        candle["Open"]
    )

    body_ratio = (
        body / candle_range
    )

    upper_wick = (
        candle["High"] -
        candle["Close"]
    ) / candle_range

    return (
        candle["Close"] >
        candle["Open"]
        and body_ratio >= 0.55
        and upper_wick <= 0.25
    )


def bearish_candle(candle):

    candle_range = (
        candle["High"] -
        candle["Low"]
    )

    if candle_range <= 0:
        return False

    body = abs(
        candle["Close"] -
        candle["Open"]
    )

    body_ratio = (
        body / candle_range
    )

    lower_wick = (
        candle["Close"] -
        candle["Low"]
    ) / candle_range

    return (
        candle["Close"] <
        candle["Open"]
        and body_ratio >= 0.55
        and lower_wick <= 0.25
    )


# ============================================================
# 📈 MARKET STRUCTURE
# ============================================================

def higher_high(df):

    if len(df) < 6:
        return False

    return (
        df["High"].iloc[-1]
        >
        df["High"].iloc[-6:-2].max()
    )


def higher_low(df):

    if len(df) < 6:
        return False

    return (
        df["Low"].iloc[-1]
        >
        df["Low"].iloc[-6:-2].min()
    )


def lower_high(df):

    if len(df) < 6:
        return False

    return (
        df["High"].iloc[-1]
        <
        df["High"].iloc[-6:-2].max()
    )


def lower_low(df):

    if len(df) < 6:
        return False

    return (
        df["Low"].iloc[-1]
        <
        df["Low"].iloc[-6:-2].min()
    )


# ============================================================
# 🧠 CREATE INDICATORS
# ============================================================

def prepare_data(df1h, df15):

    df1h = df1h.copy()
    df15 = df15.copy()

    # -------------------------------
    # 1H
    # -------------------------------

    df1h["EMA50"] = EMA(
        df1h["Close"],
        EMA_FAST
    )

    df1h["EMA200"] = EMA(
        df1h["Close"],
        EMA_TREND
    )

    df1h["RSI"] = RSI(
        df1h["Close"],
        RSI_PERIOD
    )

    df1h["ATR"] = ATR(
        df1h,
        ATR_PERIOD
    )

    df1h["VOL_MA"] = (
        df1h["Volume"]
        .rolling(VOLUME_PERIOD)
        .mean()
    )

    # Previous 20-candle high/low
    df1h["PREV_HIGH"] = (
        df1h["High"]
        .shift(1)
        .rolling(
            BREAKOUT_LOOKBACK
        )
        .max()
    )

    df1h["PREV_LOW"] = (
        df1h["Low"]
        .shift(1)
        .rolling(
            BREAKOUT_LOOKBACK
        )
        .min()
    )

    # -------------------------------
    # 15M
    # -------------------------------

    df15["EMA50"] = EMA(
        df15["Close"],
        EMA_FAST
    )

    df15["EMA200"] = EMA(
        df15["Close"],
        EMA_TREND
    )

    df15["RSI"] = RSI(
        df15["Close"],
        RSI_PERIOD
    )

    df15["ATR"] = ATR(
        df15,
        ATR_PERIOD
    )

    df15["VOL_MA"] = (
        df15["Volume"]
        .rolling(VOLUME_PERIOD)
        .mean()
    )

    return df1h, df15


# ============================================================
# 🔍 SIGNAL DETECTION
# ============================================================

def check_signal(
    df1h,
    df15,
    time_index
):

    # Need enough history
    if time_index not in df15.index:
        return None

    position = df15.index.get_loc(
        time_index
    )

    if position < 250:
        return None

    # Last closed 15M candle
    curr15 = df15.iloc[
        position
    ]

    previous_time = (
        df15.index[position - 1]
    )

    # --------------------------------------------------------
    # Find corresponding 1H candle
    # --------------------------------------------------------

    available_1h = df1h[
        df1h.index <= time_index
    ]

    if len(available_1h) < 250:
        return None

    curr1 = available_1h.iloc[-1]

    # --------------------------------------------------------
    # Build 4H
    # --------------------------------------------------------

    df4h = make_4h(
        df1h.loc[
            df1h.index <= time_index
        ]
    )

    if len(df4h) < 220:
        return None

    df4h["EMA50"] = EMA(
        df4h["Close"],
        EMA_FAST
    )

    df4h["EMA200"] = EMA(
        df4h["Close"],
        EMA_TREND
    )

    last4 = df4h.iloc[-1]

    # --------------------------------------------------------
    # MAIN TREND
    # --------------------------------------------------------

    bullish_4h = (
        last4["Close"] >
        last4["EMA200"]
        and
        last4["EMA50"] >
        last4["EMA200"]
    )

    bearish_4h = (
        last4["Close"] <
        last4["EMA200"]
        and
        last4["EMA50"] <
        last4["EMA200"]
    )

    if not bullish_4h and not bearish_4h:
        return None

    # --------------------------------------------------------
    # Previous 1H candle
    # --------------------------------------------------------

    pos1 = (
        df1h.index.get_loc(
            available_1h.index[-1]
        )
    )

    if pos1 < 3:
        return None

    prev1 = df1h.iloc[
        pos1 - 1
    ]

    # --------------------------------------------------------
    # Breakout
    # --------------------------------------------------------

    bullish_breakout = (
        curr1["Close"] >
        curr1["PREV_HIGH"]
        and
        prev1["Close"] <=
        prev1["PREV_HIGH"]
    )

    bearish_breakout = (
        curr1["Close"] <
        curr1["PREV_LOW"]
        and
        prev1["Close"] >=
        prev1["PREV_LOW"]
    )

    recent_bull = (
        df1h["Close"].iloc[
            -3:
        ]
        >
        df1h["PREV_HIGH"].iloc[
            -3:
        ]
    ).any()

    recent_bear = (
        df1h["Close"].iloc[
            -3:
        ]
        <
        df1h["PREV_LOW"].iloc[
            -3:
        ]
    ).any()

    if bullish_4h:

        if not (
            bullish_breakout
            or recent_bull
        ):
            return None

        direction = "BUY"

        breakout_level = float(
            curr1["PREV_HIGH"]
        )

    else:

        if not (
            bearish_breakout
            or recent_bear
        ):
            return None

        direction = "SELL"

        breakout_level = float(
            curr1["PREV_LOW"]
        )

    # --------------------------------------------------------
    # 15M retest
    # --------------------------------------------------------

    price = float(
        curr15["Close"]
    )

    atr = float(
        curr15["ATR"]
    )

    if atr <= 0:
        return None

    tolerance = (
        atr *
        RETEST_ATR_TOLERANCE
    )

    if direction == "BUY":

        touched = (
            curr15["Low"]
            <=
            breakout_level +
            tolerance
            and
            curr15["Low"]
            >=
            breakout_level -
            tolerance
        )

        reclaimed = (
            price >
            breakout_level
        )

    else:

        touched = (
            curr15["High"]
            >=
            breakout_level -
            tolerance
            and
            curr15["High"]
            <=
            breakout_level +
            tolerance
        )

        reclaimed = (
            price <
            breakout_level
        )

    if not touched or not reclaimed:
        return None

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = 0

    # 4H trend
    score += 2

    # 1H EMA alignment
    if direction == "BUY":

        if (
            curr1["Close"] >
            curr1["EMA200"]
            and
            curr1["EMA50"] >
            curr1["EMA200"]
        ):
            score += 1

    else:

        if (
            curr1["Close"] <
            curr1["EMA200"]
            and
            curr1["EMA50"] <
            curr1["EMA200"]
        ):
            score += 1

    # Breakout
    score += 2

    # Retest
    score += 1

    # RSI
    rsi = float(
        curr15["RSI"]
    )

    if direction == "BUY":

        if 50 <= rsi <= 72:
            score += 1

    else:

        if 28 <= rsi <= 50:
            score += 1

    # Volume
    volume_ok = (
        curr1["Volume"]
        >
        curr1["VOL_MA"] *
        VOLUME_MULTIPLIER
    )

    if volume_ok:
        score += 1

    # Candle
    if direction == "BUY":

        candle_ok = (
            bullish_candle(
                curr15
            )
        )

    else:

        candle_ok = (
            bearish_candle(
                curr15
            )
        )

    if candle_ok:
        score += 1

    # Structure
    structure_df = df1h.loc[
        :available_1h.index[-1]
    ].tail(10)

    if direction == "BUY":

        structure_ok = (
            higher_high(
                structure_df
            )
            and
            higher_low(
                structure_df
            )
        )

    else:

        structure_ok = (
            lower_high(
                structure_df
            )
            and
            lower_low(
                structure_df
            )
        )

    if structure_ok:
        score += 1

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    if score < MIN_SCORE:
        return None

    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

    entry = price

    # Add small slippage
    if direction == "BUY":

        entry *= (
            1 +
            SLIPPAGE
        )

    else:

        entry *= (
            1 -
            SLIPPAGE
        )

    # --------------------------------------------------------
    # STOP LOSS
    # --------------------------------------------------------

    if direction == "BUY":

        structural_sl = min(
            float(curr15["Low"]),
            breakout_level
        )

        sl = (
            structural_sl -
            atr *
            ATR_SL_MULTIPLIER
        )

        risk = entry - sl

        if risk <= 0:
            return None

        tp1 = entry + (
            risk * TP1_R
        )

        tp2 = entry + (
            risk * TP2_R
        )

        tp3 = entry + (
            risk * TP3_R
        )

    else:

        structural_sl = max(
            float(curr15["High"]),
            breakout_level
        )

        sl = (
            structural_sl +
            atr *
            ATR_SL_MULTIPLIER
        )

        risk = sl - entry

        if risk <= 0:
            return None

        tp1 = entry - (
            risk * TP1_R
        )

        tp2 = entry - (
            risk * TP2_R
        )

        tp3 = entry - (
            risk * TP3_R
        )

    return {
        "time": time_index,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "risk": risk,
        "score": score,
        "rsi": rsi
    }


# ============================================================
# 🎯 SIMULATE TRADE
# ============================================================

def simulate_trade(
    df15,
    signal,
    start_position
):

    direction = signal["direction"]

    entry = signal["entry"]
    sl = signal["sl"]
    tp1 = signal["tp1"]
    tp2 = signal["tp2"]
    tp3 = signal["tp3"]

    end_position = min(
        start_position +
        MAX_TRADE_BARS,
        len(df15) - 1
    )

    for i in range(
        start_position + 1,
        end_position + 1
    ):

        candle = df15.iloc[i]

        high = float(
            candle["High"]
        )

        low = float(
            candle["Low"]
        )

        # ====================================================
        # BUY
        # ====================================================

        if direction == "BUY":

            # Conservative assumption:
            # if TP and SL are both touched
            # in the same candle,
            # assume SL happened first.

            if low <= sl:

                return {
                    "result": "LOSS",
                    "R": -1.0,
                    "exit": sl,
                    "exit_time": df15.index[i]
                }

            if high >= tp3:

                return {
                    "result": "WIN",
                    "R": TP3_R,
                    "exit": tp3,
                    "exit_time": df15.index[i]
                }

            if high >= tp2:

                return {
                    "result": "WIN",
                    "R": TP2_R,
                    "exit": tp2,
                    "exit_time": df15.index[i]
                }

            if high >= tp1:

                return {
                    "result": "WIN",
                    "R": TP1_R,
                    "exit": tp1,
                    "exit_time": df15.index[i]
                }

        # ====================================================
        # SELL
        # ====================================================

        else:

            if high >= sl:

                return {
                    "result": "LOSS",
                    "R": -1.0,
                    "exit": sl,
                    "exit_time": df15.index[i]
                }

            if low <= tp3:

                return {
                    "result": "WIN",
                    "R": TP3_R,
                    "exit": tp3,
                    "exit_time": df15.index[i]
                }

            if low <= tp2:

                return {
                    "result": "WIN",
                    "R": TP2_R,
                    "exit": tp2,
                    "exit_time": df15.index[i]
                }

            if low <= tp1:

                return {
                    "result": "WIN",
                    "R": TP1_R,
                    "exit": tp1,
                    "exit_time": df15.index[i]
                }

    # ========================================================
    # TIME EXIT
    # ========================================================

    last = df15.iloc[
        end_position
    ]

    exit_price = float(
        last["Close"]
    )

    if direction == "BUY":

        raw_r = (
            exit_price -
            entry
        ) / (
            entry -
            sl
        )

    else:

        raw_r = (
            entry -
            exit_price
        ) / (
            sl -
            entry
        )

    return {
        "result": "TIME_EXIT",
        "R": raw_r,
        "exit": exit_price,
        "exit_time": df15.index[
            end_position
        ]
    }


# ============================================================
# 💰 APPLY FEES
# ============================================================

def apply_fees(r_value):

    # Approximate round-trip fee
    fee_r = (
        FEE_RATE * 2
    )

    return r_value - fee_r


# ============================================================
# 📊 BACKTEST ONE SYMBOL
# ============================================================

def backtest_symbol(
    symbol,
    df1h,
    df15
):

    print(
        f"\n🔬 Backtesting {symbol}"
    )

    df1h, df15 = prepare_data(
        df1h,
        df15
    )

    trades = []

    start_position = 250

    last_trade_end = -1

    for i in range(
        start_position,
        len(df15) - 1
    ):

        # Avoid overlapping trades
        if i <= last_trade_end:
            continue

        timestamp = df15.index[i]

        signal = check_signal(
            df1h,
            df15,
            timestamp
        )

        if signal is None:
            continue

        result = simulate_trade(
            df15,
            signal,
            i
        )

        if result is None:
            continue

        final_r = apply_fees(
            result["R"]
        )

        trade = {
            "symbol": symbol,
            "entry_time": signal["time"],
            "direction": signal["direction"],
            "score": signal["score"],
            "entry": signal["entry"],
            "sl": signal["sl"],
            "tp1": signal["tp1"],
            "tp2": signal["tp2"],
            "tp3": signal["tp3"],
            "exit": result["exit"],
            "exit_time": result["exit_time"],
            "result": result["result"],
            "R": final_r
        }

        trades.append(
            trade
        )

        last_trade_end = (
            df15.index.get_loc(
                result["exit_time"]
            )
        )

    return trades


# ============================================================
# 📈 STATISTICS
# ============================================================

def calculate_statistics(
    trades
):

    if not trades:

        return {
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "Win Rate %": 0,
            "Profit Factor": 0,
            "Net R": 0,
            "Average R": 0,
            "Max Drawdown R": 0,
            "Max Consecutive Losses": 0
        }

    r_values = [
        float(
            t["R"]
        )
        for t in trades
    ]

    wins = [
        r
        for r in r_values
        if r > 0
    ]

    losses = [
        r
        for r in r_values
        if r < 0
    ]

    total_trades = len(
        r_values
    )

    win_count = len(
        wins
    )

    loss_count = len(
        losses
    )

    win_rate = (
        win_count /
        total_trades *
        100
    )

    gross_profit = sum(
        wins
    )

    gross_loss = abs(
        sum(losses)
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit /
            gross_loss
        )

    else:

        profit_factor = float(
            "inf"
        )

    net_r = sum(
        r_values
    )

    average_r = (
        net_r /
        total_trades
    )

    # ========================================================
    # EQUITY CURVE
    # ========================================================

    equity = 0

    peak = 0

    max_drawdown = 0

    current_losses = 0

    max_consecutive_losses = 0

    for r in r_values:

        equity += r

        peak = max(
            peak,
            equity
        )

        drawdown = (
            peak -
            equity
        )

        max_drawdown = max(
            max_drawdown,
            drawdown
        )

        if r < 0:

            current_losses += 1

            max_consecutive_losses = max(
                max_consecutive_losses,
                current_losses
            )

        else:

            current_losses = 0

    return {
        "Trades": total_trades,
        "Wins": win_count,
        "Losses": loss_count,
        "Win Rate %": round(
            win_rate,
            2
        ),
        "Profit Factor": round(
            profit_factor,
            3
        ),
        "Net R": round(
            net_r,
            3
        ),
        "Average R": round(
            average_r,
            3
        ),
        "Max Drawdown R": round(
            max_drawdown,
            3
        ),
        "Max Consecutive Losses":
            max_consecutive_losses
    }


# ============================================================
# 🚀 MAIN BACKTEST
# ============================================================

def main():

    print("=" * 75)

    print(
        "🚀 CRYPTO STRATEGY BACKTEST V5"
    )

    print(
        "4H TREND + 1H BREAKOUT + "
        "15M RETEST"
    )

    print(
        f"📅 Backtest period: "
        f"{BACKTEST_DAYS} days"
    )

    print(
        f"💰 Initial balance: "
        f"${INITIAL_BALANCE:,.2f}"
    )

    print(
        f"⚠️ Risk per trade: "
        f"{RISK_PERCENT}%"
    )

    print("=" * 75)

    end_dt = datetime.now(
        timezone.utc
    )

    start_dt = (
        end_dt -
        pd.Timedelta(
            days=BACKTEST_DAYS
        )
    )

    start_ms = int(
        start_dt.timestamp() *
        1000
    )

    end_ms = int(
        end_dt.timestamp() *
        1000
    )

    all_trades = []

    symbol_results = []

    for index, symbol in enumerate(
        SYMBOLS,
        1
    ):

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"[{index}/{len(SYMBOLS)}] "
            f"{symbol}"
        )

        # ----------------------------------------------------
        # Download 1H
        # ----------------------------------------------------

        df1h = get_klines(
            symbol,
            TF_1H,
            start_ms,
            end_ms
        )

        if df1h is None:

            print(
                "❌ No 1H data"
            )

            continue

        # ----------------------------------------------------
        # Download 15M
        # ----------------------------------------------------

        df15 = get_klines(
            symbol,
            TF_15M,
            start_ms,
            end_ms
        )

        if df15 is None:

            print(
                "❌ No 15M data"
            )

            continue

        print(
            f"📊 1H candles: "
            f"{len(df1h)}"
        )

        print(
            f"📊 15M candles: "
            f"{len(df15)}"
        )

        # ----------------------------------------------------
        # Backtest
        # ----------------------------------------------------

        trades = backtest_symbol(
            symbol,
            df1h,
            df15
        )

        all_trades.extend(
            trades
        )

        stats = calculate_statistics(
            trades
        )

        symbol_results.append(
            {
                "Symbol": symbol,
                **stats
            }
        )

        print(
            f"📈 Trades: "
            f"{stats['Trades']}"
        )

        print(
            f"🎯 Win Rate: "
            f"{stats['Win Rate %']}%"
        )

        print(
            f"💰 Profit Factor: "
            f"{stats['Profit Factor']}"
        )

        print(
            f"📊 Net R: "
            f"{stats['Net R']}"
        )

        print(
            f"📉 Max DD: "
            f"{stats['Max Drawdown R']}R"
        )

        time.sleep(
            0.5
        )

    # ========================================================
    # SAVE TRADES
    # ========================================================

    if all_trades:

        trades_df = pd.DataFrame(
            all_trades
        )

        trades_df.to_csv(
            "backtest_trades.csv",
            index=False
        )

        print(
            "\n💾 Saved: "
            "backtest_trades.csv"
        )

    # ========================================================
    # SAVE SYMBOL RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        symbol_results
    )

    if not results_df.empty:

        results_df = results_df.sort_values(
            by="Net R",
            ascending=False
        )

        results_df.to_csv(
            "backtest_results.csv",
            index=False
        )

        print(
            "💾 Saved: "
            "backtest_results.csv"
        )

    # ========================================================
    # GLOBAL RESULTS
    # ========================================================

    global_stats = calculate_statistics(
        all_trades
    )

    print(
        "\n\n"
        + "=" * 75
    )

    print(
        "🏆 GLOBAL BACKTEST RESULTS"
    )

    print(
        "=" * 75
    )

    for key, value in global_stats.items():

        print(
            f"{key}: {value}"
        )

    # ========================================================
    # BUY / SELL
    # ========================================================

    if all_trades:

        buy_trades = [
            t for t in all_trades
            if t["direction"] == "BUY"
        ]

        sell_trades = [
            t for t in all_trades
            if t["direction"] == "SELL"
        ]

        buy_stats = calculate_statistics(
            buy_trades
        )

        sell_stats = calculate_statistics(
            sell_trades
        )

        print(
            "\n"
            + "=" * 75
        )

        print(
            "🟢 BUY RESULTS"
        )

        print(
            "=" * 75
        )

        for key, value in buy_stats.items():

            print(
                f"{key}: {value}"
            )

        print(
            "\n"
            + "=" * 75
        )

        print(
            "🔴 SELL RESULTS"
        )

        print(
            "=" * 75
        )

        for key, value in sell_stats.items():

            print(
                f"{key}: {value}"
            )

    # ========================================================
    # TOP 10
    # ========================================================

    if not results_df.empty:

        print(
            "\n"
            + "=" * 75
        )

        print(
            "🥇 TOP 10 SYMBOLS"
        )

        print(
            "=" * 75
        )

        print(
            results_df.head(10).to_string(
                index=False
            )
        )

        print(
            "\n"
            + "=" * 75
        )

        print(
            "💀 WORST 10 SYMBOLS"
        )

        print(
            "=" * 75
        )

        print(
            results_df.tail(10).sort_values(
                by="Net R"
            ).to_string(
                index=False
            )
        )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "✅ BACKTEST COMPLETED"
    )

    print(
        "=" * 75
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
