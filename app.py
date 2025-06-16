
import streamlit as st
import pandas as pd
from fetch_data import get_btc_data
from technicals import add_technicals
from sentiment import get_sentiment_score, analyze_whale_sentiment
from train_model import train_and_predict, generate_score
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Trend Predictor", layout="wide")
st.title("📊 Crypto Trend Predictor")

try:
    df = get_btc_data()
    if df.empty or len(df) < 20:
        st.error("🚫 Not enough data to make a prediction. Please check your source or wait for market activity.")
    else:
        df = add_technicals(df)
        model, report = train_and_predict(df)
        score = generate_score(report)

        st.subheader("📈 BTC Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['price'], mode='lines', name='BTC Price'))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧠 Sentiment + Whale Activity")
        sentiment_score = get_sentiment_score()
        whale_summary = analyze_whale_sentiment()

        st.metric("Sentiment Score", sentiment_score)
        st.write("📋 Whale Insight Summary:")
        st.info(whale_summary)

        st.subheader("🤖 Predictor Meter")
        if score >= 80:
            verdict = "Strong Buy"
        elif score >= 60:
            verdict = "Buy"
        elif score >= 50:
            verdict = "Neutral"
        else:
            verdict = "Avoid"

        st.progress(score)
        st.success(f"Model Score: {score} — Recommendation: **{verdict}**")

except Exception as e:
    st.error(f"❌ Failed to fetch BTC data or model error: {e}")
