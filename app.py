import streamlit as st
from fetch_data import get_btc_data
from technicals import add_technicals
from sentiment import get_sentiment_score, analyze_whale_sentiment
from train_model import train_and_predict, generate_score
import plotly.graph_objects as go

st.title("📊 Crypto Trend Predictor")


df = get_btc_data()
if df is None or df.empty or len(df) < 10:
    st.error("🚫 Not enough data to make a prediction. Please check your data source.")
    st.stop()

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

st.subheader("📍 Predictor Meter (Should You Invest?)")
score = generate_score(df, avg_sentiment, whale_sentiment)

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score,
    title={'text': "Investment Recommendation Score"},
    gauge={
        'axis': {'range': [-4, 4]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [-4, -2], 'color': "red"},
            {'range': [-2, 0], 'color': "orange"},
            {'range': [0, 2], 'color': "yellow"},
            {'range': [2, 4], 'color': "green"}
        ],
    }
))
st.plotly_chart(gauge)
