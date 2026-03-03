import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# Commodity ticker configuration (continuous front-month)
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

# Futures contract config for real forward curves
FUTURES_CONTRACT_CONFIG = {
    "Brent Crude":  {"root": "BZ", "suffix": ".NYM", "monthly": True},
    "WTI Crude":    {"root": "CL", "suffix": ".NYM", "monthly": True},
    "Natural Gas":  {"root": "NG", "suffix": ".NYM", "monthly": True},
    "Gold":         {"root": "GC", "suffix": ".CMX", "monthly": [2, 4, 6, 8, 12]},
    "Silver":       {"root": "SI", "suffix": ".CMX", "monthly": [3, 5, 7, 9, 12]},
    "Corn":         {"root": "ZC", "suffix": ".CBT", "monthly": [3, 5, 7, 9, 12]},
    "Soybean":      {"root": "ZS", "suffix": ".CBT", "monthly": [1, 3, 5, 7, 8, 9, 11]},
    "Copper":       {"root": "HG", "suffix": ".CMX", "monthly": True}
}

MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"
}

# --- Forward Curve ---

def _build_future_tickers(commodity_name, n_months=6):
    """Generate Yahoo Finance ticker symbols for the next N contract months."""
    config = FUTURES_CONTRACT_CONFIG.get(commodity_name)
    if not config:
        return []
    root, suffix, monthly = config["root"], config["suffix"], config["monthly"]
    today = datetime.now()
    tickers = []
    check_date = today.replace(day=1) + timedelta(days=32)
    check_date = check_date.replace(day=1)
    attempts = 0
    while len(tickers) < n_months and attempts < 24:
        month = check_date.month
        year_short = check_date.year % 100
        if monthly is True or month in monthly:
            code = MONTH_CODES[month]
            ticker = f"{root}{code}{year_short:02d}{suffix}"
            label = check_date.strftime("%b %Y")
            tickers.append({"ticker": ticker, "label": label, "date": check_date})
        check_date = (check_date + timedelta(days=32)).replace(day=1)
        attempts += 1
    return tickers

@st.cache_data(ttl=900)
def get_forward_curve(commodity_name):
    """Fetch actual futures strip prices. Cached for 15 min."""
    contracts = _build_future_tickers(commodity_name, n_months=6)
    if not contracts:
        return None
    curve = []
    for c in contracts:
        try:
            data = yf.Ticker(c["ticker"]).history(period="5d")
            if not data.empty:
                price = float(data['Close'].iloc[-1])
                if price > 0:
                    curve.append({"label": c["label"], "price": price,
                                  "ticker": c["ticker"], "date": c["date"]})
        except Exception:
            continue
    return curve if len(curve) >= 2 else None

def get_front_month_price(commodity_name):
    """Get the front-month futures price. Default hedge price."""
    curve = get_forward_curve(commodity_name)
    if curve:
        return curve[0]["price"]
    return None

def get_matched_forward_price(commodity_name, voyage_days):
    """Get the forward price for the contract month closest to the voyage arrival date.
    This is the contract a trader would sell to hedge a cargo in transit."""
    curve = get_forward_curve(commodity_name)
    if not curve:
        return None, None
    arrival = datetime.now() + timedelta(days=voyage_days)
    best = None
    for point in curve:
        if point["date"] >= arrival:
            best = point
            break
    if best is None:
        best = curve[-1]
    return best["price"], best["label"]

# --- Market Data ---

def get_market_data(ticker_symbol):
    """Fetch spot prices, returns, volatility, drift, and kurtosis."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="6mo")
        if df.empty:
            return None
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
            "history": close,
            "kurtosis": returns.tail(60).kurtosis() if len(returns) >= 60 else 0.0
        }
    except Exception as e:
        print(f"Market Data Error: {e}")
        return None

# --- Seasonality ---

@st.cache_data(ttl=3600)
def get_seasonality_data(ticker_symbol):
    """Fetch 10Y of daily data. Return monthly avg, min, max + current year."""
    try:
        df = yf.Ticker(ticker_symbol).history(period="10y")
        if df.empty or len(df) < 252:
            return None
        df = df[['Close']].copy()
        df['month'] = df.index.month
        df['year'] = df.index.year
        current_year = datetime.now().year
        monthly = df.groupby(['year', 'month'])['Close'].mean().reset_index()
        hist = monthly[monthly['year'] < current_year]
        agg = hist.groupby('month')['Close'].agg(['mean', 'min', 'max'])
        cy = monthly[monthly['year'] == current_year]
        cy_series = cy.set_index('month')['Close'] if not cy.empty else None
        return {
            "avg": agg['mean'].values,
            "min": agg['min'].values,
            "max": agg['max'].values,
            "current_year": cy_series,
            "current_year_label": current_year
        }
    except Exception:
        return None

# --- Fuel & Carbon ---

def get_fuel_market_price():
    """Fetch Heating Oil as proxy for MGO, convert Gallons to Metric Tons."""
    try:
        price_gal = yf.Ticker("HO=F").history(period="1d")['Close'].iloc[-1]
        return float(price_gal * 312)
    except:
        return 650.0

def get_carbon_price():
    """Fetch European Carbon Price (EUA) via KEUA ETF proxy."""
    try:
        hist = yf.Ticker("KEUA").history(period="1d")
        return float(hist['Close'].iloc[-1]) if not hist.empty else 75.0
    except:
        return 75.0

# --- Correlations (1wk, 1mo, 1y, 10y) ---

PERIOD_MAP = {
    "1wk": {"download_period": "3mo", "tail_days": 5},
    "1mo": {"download_period": "3mo", "tail_days": 21},
    "1y":  {"download_period": "2y", "tail_days": 252},
    "10y": {"download_period": "10y", "tail_days": 2520},
}

@st.cache_data(ttl=1800)
def _fetch_corr_data(download_period):
    """Download and cache correlation data."""
    try:
        data = yf.download(list(COMMODITIES.values()), period=download_period, progress=False)['Close']
        return data
    except:
        return None

def get_correlations(target_ticker, period="1mo"):
    """Correlation matrix between target and other commodities."""
    cfg = PERIOD_MAP.get(period, PERIOD_MAP["1mo"])
    try:
        data = _fetch_corr_data(cfg["download_period"])
        if data is None:
            return pd.DataFrame()
        returns = data.pct_change()
        returns = returns.tail(cfg["tail_days"])
        corr = returns.corr()
        if target_ticker not in corr.columns:
            return pd.DataFrame()
        target_series = corr[target_ticker].drop(target_ticker).to_frame(name="Correlation")
        target_series['Product'] = target_series.index.map(TICKER_TO_NAME)
        return target_series.sort_values(by='Correlation', ascending=False)[['Product', 'Correlation']]
    except:
        return pd.DataFrame()

# --- Macro ---

def get_macro_data():
    """Fetch 13-week Treasury Bill rate as proxy for SOFR."""
    try:
        data = yf.download("^IRX", period="5d", progress=False)['Close']
        return {"sofr_proxy": float(data.iloc[-1])} if not data.empty else {"sofr_proxy": 5.35}
    except:
        return {"sofr_proxy": 5.35}
