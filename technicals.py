import ta

def add_technicals(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_diff'] = macd.macd_diff()
    return df.dropna()
