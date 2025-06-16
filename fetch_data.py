import yfinance as yf

def get_btc_data(period="60d", interval="1d"):
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period=period, interval=interval)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df
