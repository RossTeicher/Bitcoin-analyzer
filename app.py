import streamlit as st
from fetch_data import get_btc_data
from technicals import add_technicals
from sentiment import get_sentiment_score, analyze_whale_sentiment
from train_model import train_and_predict
import matplotlib.pyplot as plt

st.title("📊 Crypto Trend Predictor")

df = get_btc_data()
df = add_technicals(df)

st.subheader("Bitcoin Price (Last 60 Days)")
st.line_chart(df['Close'])

st.subheader("Technical Indicators")
st.line_chart(df[['rsi', 'macd', 'macd_diff']])

st.subheader("Sentiment Analysis")
news = st.text_area("Paste crypto news headlines (one per line)", 
'''
BlackRock receives approval for spot Bitcoin ETF
SEC cracks down on unregistered crypto exchanges
''')

news_list = [line.strip() for line in news.strip().split("\n") if line]
sentiment_scores = [get_sentiment_score(h) for h in news_list]
avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

st.write("Average News Sentiment Score:", avg_sentiment)

whale_alerts = [
    "Whale sends 2,000 BTC to Binance",
    "1,000 BTC moved from Coinbase to cold wallet"
]
whale_sentiment = analyze_whale_sentiment(whale_alerts)
st.write("Whale Sentiment Score:", whale_sentiment)

st.subheader("Model Prediction")
model = train_and_predict(df)
last_row = df.iloc[-1][['rsi', 'macd', 'macd_diff']].values.reshape(1, -1)
prediction = model.predict(last_row)[0]

st.markdown(f"### 🚦 Predicted Trend: {'📈 Bullish' if prediction == 1 else '📉 Bearish'}")
