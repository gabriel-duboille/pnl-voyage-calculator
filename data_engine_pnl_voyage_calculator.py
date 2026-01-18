import yfinance as yf
import pandas as pd
import numpy as np

# Commodity ticker configuration
COMMODITIES = {
    "Brent Crude": "BZ=F",
    "WTI Crude": "CL=F",
    "Natural Gas": "NG=F",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Corn": "ZC=F",
    "Soybean": "ZS=F",
    "Copper": "HG=F"
}

TICKER_TO_NAME = {v: k for k, v in COMMODITIES.items()}

def get_market_data(ticker_symbol):
    """Fetch spot prices, returns, volatility, and drift for a given ticker."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="6mo")
        if df.empty: return None

        close = df['Close']
        returns = close.pct_change().dropna()
        
        return {
            "price": close.iloc[-1],
            "delta_1d": close.pct_change(1).iloc[-1],
            "delta_1w": close.pct_change(5).iloc[-1] if len(df) > 5 else 0,
            "delta_1m": close.pct_change(21).iloc[-1] if len(df) > 21 else 0,
            "volatility": returns.tail(21).std() * np.sqrt(252),
            "drift_annual": returns.tail(30).mean() * 252,
            "momentum_30d": returns.tail(21).sum(),
            "history": close
        }
    except Exception as e:
        print(f"Market Data Error: {e}")
        return None

def get_fuel_market_price():
    """Fetch Heating Oil as a proxy for MGO and convert from Gallons to Metric Tons."""
    try:
        # HO=F is quoted in USD per gallon. ~312 gallons per metric ton.
        price_gal = yf.Ticker("HO=F").history(period="1d")['Close'].iloc[-1]
        return float(price_gal * 312) 
    except:
        return 650.0 # Standard maritime industry fallback

def get_carbon_price():
    """Fetch European Carbon Price (EUA) via KEUA ETF proxy."""
    try:
        hist = yf.Ticker("KEUA").history(period="1d")
        return float(hist['Close'].iloc[-1]) if not hist.empty else 75.0
    except:
        return 75.0

def get_correlations(target_ticker, period="1mo"):
    """Calculate correlation matrix between the target asset and other commodities."""
    try:
        data = yf.download(list(COMMODITIES.values()), period="3mo", progress=False)['Close']
        returns = data.pct_change()
        returns = returns.tail(5) if period == "1wk" else returns.tail(21)
        
        corr = returns.corr()
        if target_ticker not in corr.columns: return pd.DataFrame()
        
        target_series = corr[target_ticker].drop(target_ticker).to_frame(name="Correlation")
        target_series['Product'] = target_series.index.map(TICKER_TO_NAME)
        return target_series.sort_values(by='Correlation', ascending=False)[['Product', 'Correlation']]
    except:
        return pd.DataFrame()

def get_macro_data():
    """Fetch 13-week Treasury Bill rate as a proxy for SOFR/risk-free rates."""
    try:
        data = yf.download("^IRX", period="5d", progress=False)['Close']
        return {"sofr_proxy": float(data.iloc[-1])} if not data.empty else {"sofr_proxy": 5.35}
    except:
        return {"sofr_proxy": 5.35}