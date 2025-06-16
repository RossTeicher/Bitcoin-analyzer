
import yfinance as yf
import pandas as pd
import streamlit as st

def get_btc_data():
    try:
        df = yf.download("BTC-USD", period="14d", interval="1h")
        if df is not None and not df.empty:
            df = df.dropna()
            df['Return'] = df['Close'].pct_change()
            df = df.dropna()
            st.write("✅ Data loaded:", len(df), "rows")
            st.dataframe(df.tail(5))
            return df
        else:
            st.error("❌ Failed to fetch BTC data.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Exception while fetching data: {e}")
        return pd.DataFrame()
