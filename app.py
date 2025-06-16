
import streamlit as st
from fetch_data import get_btc_data
from technicals import add_technicals
from sentiment import get_sentiment_score, analyze_whale_sentiment
from train_model import train_and_predict, generate_score
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Trend Predictor", layout="wide")
st.title("📊 Crypto Trend Predictor")

# --- DATA FETCHING ---
df = get_btc_data()
if df is None or df.empty or len(df) < 10:
    st.error("🚫 Not enough data to make a prediction. Please check your source or wait for market activity.")
    st.stop()

# --- MODEL TRAINING ---
model = train_and_predict(df)

# --- LIVE PRICE ---
btc = yf.Ticker("BTC-USD")
eth = yf.Ticker("ETH-USD")
btc_price = btc.history(period="1d").iloc[-1]['Close']
eth_price = eth.history(period="1d").iloc[-1]['Close']
st.subheader("💰 Live Prices")
st.metric(label="Bitcoin (BTC)", value=f"${btc_price:,.2f}")
st.metric(label="Ethereum (ETH)", value=f"${eth_price:,.2f}")

# --- HISTORICAL CHART ---
hist = btc.history(period="7d", interval="1h")
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='BTC Price'))
fig.update_layout(title='📈 Bitcoin 7-Day Trend', xaxis_title='Time', yaxis_title='Price (USD)')
st.plotly_chart(fig)


# --- WHALE ALERTS ---
st.subheader("🐋 Whale Alerts")
whales = analyze_whale_sentiment()
for whale in whales:
    st.write(f"Whale {whale['id']}: {whale['type']} {whale['amount']} {whale['coin']}")

# --- SENTIMENT HEATMAP ---
import plotly.express as px
st.subheader("🔥 Sentiment Heatmap (24h)")
heatmap_df = generate_sentiment_heatmap()
fig2 = px.imshow([heatmap_df['Sentiment']], 
                 labels=dict(x="Hour", color="Sentiment Score"),
                 x=heatmap_df["Hour"],
                 y=["Score"],
                 aspect="auto")
st.plotly_chart(fig2)

score = generate_score()
st.subheader("🎯 Should I Buy?")
if score > 0.7:
    st.success("😎 Strong Buy Signal")
elif score > 0.4:
    st.info("🤔 Neutral to Cautious")
else:
    st.warning("⚠️ Not a Good Entry Right Now")
