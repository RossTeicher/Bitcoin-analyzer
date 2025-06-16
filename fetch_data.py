
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

def get_btc_data():
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {
            "vs_currency": "usd",
            "days": "90",
            "interval": "daily"
        }
        response = requests.get(url, params=params)
        data = response.json()
        prices = data["prices"]

        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df["Return"] = df["price"].pct_change()
        df.dropna(inplace=True)

        if df.empty:
            st.error("❌ CoinGecko returned no data.")
            return pd.DataFrame()

        st.success(f"✅ Loaded {len(df)} rows from CoinGecko")
        st.dataframe(df.tail(5))
        return df
    except Exception as e:
        st.error(f"❌ Error fetching CoinGecko data: {e}")
        return pd.DataFrame()
