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

   
