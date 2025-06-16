
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

def prepare_data(df):
    df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
    df.dropna(inplace=True)
    X = df[['price', 'Return']]
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_predict(df):
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    return model, report
