from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

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
