from transformers import pipeline
import re

sentiment_pipeline = pipeline("sentiment-analysis")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.strip()

def get_sentiment_score(text):
    result = sentiment_pipeline(clean_text(text))[0]
    return 1 if result["label"] == "POSITIVE" else -1

def analyze_whale_sentiment(alerts):
    scores = [get_sentiment_score(alert) for alert in alerts]
    return sum(scores) / len(scores)
