
import random
import pandas as pd

def get_sentiment_score():
    # Simulate a sentiment score for now
    return random.uniform(0.3, 0.9)

def analyze_whale_sentiment():
    # Simulate whale alerts using random large transactions
    whale_alerts = []
    for i in range(random.randint(2, 5)):
        whale_alerts.append({
            "id": i,
            "coin": "BTC",
            "type": random.choice(["Buy", "Sell"]),
            "amount": random.randint(1000, 10000)
        })
    return whale_alerts

def generate_sentiment_heatmap():
    hours = [f"{i}:00" for i in range(24)]
    sentiment_data = [random.uniform(0.2, 0.9) for _ in range(24)]
    return pd.DataFrame({"Hour": hours, "Sentiment": sentiment_data})
