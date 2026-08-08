import os
import requests
import pandas as pd

BASE = "https://fapi.binance.com"
TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

INTERVAL = "1h"
LIMIT = 220
CONFIRM_WINDOW = 3  # RSI event and MACD cross must occur within 3 closed 1H candles

def get(path, params=None):
    r = requests.get(BASE + path, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_symbols():
    data = get("/fapi/v1/exchangeInfo")
    return [
        s["symbol"] for s in data["symbols"]
        if s["status"] == "TRADING"
        and s["quoteAsset"] == "USDT"
        and s["contractType"] == "PERPETUAL"
    ]

def analyze(symbol):
    data = get("/fapi/v1/klines", {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": LIMIT
    })

    if len(data) < 210:
        return None

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    # Ignore the currently forming candle.
    close = pd.to_numeric(df["close"]).iloc[:-1].reset_index(drop=True)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    # Events on closed candles:
    # RSI BUY = RSI was below 30 and then crossed upward through 30.
    # RSI SELL = RSI was above 70 and then crossed downward through 70.
    rsi_buy = (rsi.shift(1) < 30) & (rsi >= 30)
    rsi_sell = (rsi.shift(1) > 70) & (rsi <= 70)

    # MACD BUY/SELL = actual MACD/Signal cross.
    macd_buy = (macd.shift(1) <= signal.shift(1)) & (macd > signal)
    macd_sell = (macd.shift(1) >= signal.shift(1)) & (macd < signal)

    recent_rsi_buy = rsi_buy.iloc[-CONFIRM_WINDOW:].any()
    recent_macd_buy = macd_buy.iloc[-CONFIRM_WINDOW:].any()
    recent_rsi_sell = rsi_sell.iloc[-CONFIRM_WINDOW:].any()
    recent_macd_sell = macd_sell.iloc[-CONFIRM_WINDOW:].any()

    current_rsi = float(rsi.iloc[-1])

    if recent_rsi_buy and recent_macd_buy:
        return ("BUY", current_rsi)

    if recent_rsi_sell and recent_macd_sell:
        return ("SELL", current_rsi)

    return None

def send_telegram(text):
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": text},
        timeout=20
    )
    r.raise_for_status()

def main():
    buys = []
    sells = []

    for symbol in get_symbols():
        try:
            result = analyze(symbol)
            if result:
                side, rsi = result
                if side == "BUY":
                    buys.append(f"🟢 {symbol} | RSI {rsi:.1f}")
                else:
                    sells.append(f"🔴 {symbol} | RSI {rsi:.1f}")
        except Exception as e:
            print(f"{symbol}: {e}")

    lines = ["📊 BINANCE FUTURES — 1H", "", "شروط الإشارة: RSI + MACD", ""]

    if buys:
        lines.append("🟢 BUY")
        lines.extend(buys)
        lines.append("")

    if sells:
        lines.append("🔴 SELL")
        lines.extend(sells)
        lines.append("")

    if not buys and not sells:
        lines.append("لا توجد إشارة مطابقة حاليًا.")

    message = "\n".join(lines)
    send_telegram(message)
    print(message)

if __name__ == "__main__":
    main()
