from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def prepare_data(df):
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    X = df[['rsi', 'macd', 'macd_diff']]
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_predict(df):
    X_train, X_test, y_train, y_test = prepare_data(df)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return clf

def generate_score(df, news_sentiment, whale_sentiment):
    score = 0
    latest = df.iloc[-1]

    if latest['rsi'] < 30:
        score += 1
    elif latest['rsi'] > 70:
        score -= 1

    if latest['macd'] > 0:
        score += 1
    else:
        score -= 1

    if news_sentiment > 0.3:
        score += 1
    elif news_sentiment < -0.3:
        score -= 1

    if whale_sentiment > 0.3:
        score += 1
    elif whale_sentiment < -0.3:
        score -= 1

    return max(-4, min(4, score))  # Clamp between -4 and +4
