# Binance Futures RSI + MACD Scanner

- Binance USDT-M perpetual futures
- 1H timeframe
- Uses only CLOSED candles
- BUY: RSI exits oversold (<30 -> >=30) AND MACD crosses above Signal within 3 closed candles
- SELL: RSI exits overbought (>70 -> <=70) AND MACD crosses below Signal within 3 closed candles
- Runs hourly with GitHub Actions
- No Binance API key required
