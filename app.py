
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Crypto Predictor", layout="wide")

# Load data
df = pd.read_csv("btc_data.csv")
df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df.dropna(inplace=True)

# Features and target
X = df[["Return", "whale_trend"]]
y = df["target"]

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
pred = model.predict(X_test)[-1]

# Visualization
st.title("📊 Crypto Trend Predictor")
st.plotly_chart(go.Figure(go.Scatter(x=df['date'], y=df['Close'], mode='lines', name='BTC Close Price')))

# Investment meter
st.subheader("Should I invest?")
if pred == 1:
    st.success("✅ YES — Our model suggests upward movement.")
else:
    st.error("🚫 NO — Our model suggests caution.")

st.markdown("---")
st.caption("Note: This tool is for educational purposes only.")
