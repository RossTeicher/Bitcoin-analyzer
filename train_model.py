
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

def generate_score(report):
    # Use weighted avg f1-score as basis for score
    try:
        f1 = report["weighted avg"]["f1-score"]
        if f1 > 0.8:
            return 90
        elif f1 > 0.7:
            return 75
        elif f1 > 0.6:
            return 60
        elif f1 > 0.5:
            return 50
        else:
            return 30
    except:
        return 0
