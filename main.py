from src.fetch_data import get_btc_data
from src.technicals import add_technicals
from src.sentiment import get_sentiment_score, analyze_whale_sentiment
from src.train_model import train_and_predict

# Get BTC Data
df = get_btc_data()
df = add_technicals(df)

# Simulate news + whale activity
headlines = [
    "BlackRock receives approval for spot Bitcoin ETF",
    "SEC cracks down on unregistered crypto exchanges"
]
headline_sentiment = sum([get_sentiment_score(h) for h in headlines]) / len(headlines)

whale_sentiment = analyze_whale_sentiment([
    "Whale sends 2,000 BTC to Binance",
    "1,000 BTC moved from Coinbase to cold wallet"
])

# Print sentiment scores
print(f"Headline Sentiment: {headline_sentiment}")
print(f"Whale Sentiment: {whale_sentiment}")

# Train Model
train_and_predict(df)
