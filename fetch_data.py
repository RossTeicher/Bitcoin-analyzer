
import requests
import pandas as pd
from datetime import datetime

def get_btc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "30", "interval": "hourly"}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        return pd.DataFrame()

    data = response.json()
    prices = data.get("prices", [])

    if not prices:
        return pd.DataFrame()

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.rename(columns={"timestamp": "date"}, inplace=True)
    df["Close"] = df["price"]  # Ensure compatibility with model expecting 'Close'
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    return df
