import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pytz
from datetime import datetime
import time
import plotly.express as px

# --- 1. SETTINGS ---
st.set_page_config(page_title="PRO AI Stock Engine", layout="wide")
IST = pytz.timezone('Asia/Kolkata')

# --- 2. UPDATED WATCHLIST (Top 20 Mid Caps) ---
watchlist = [
    "FEDERALBNK.NS", "IDFCFIRSTB.NS", "L&TFH.NS", "GMRINFRA.NS", "RECLTD.NS", 
    "PFC.NS", "NATIONALUM.NS", "SAIL.NS", "NMDC.NS", "OIL.NS", 
    "MANAPPURAM.NS", "ABCAPITAL.NS", "BANDHANBNK.NS", "VOLTAS.NS", 
    "ASHOKLEY.NS", "CONCOR.NS", "PETRONET.NS", "TATACHEMICALS.NS", "AUBANK.NS", "POLYCAB.NS"
]

# --- 3. ROBUST FETCH FUNCTION ---
def fetch_with_retry(ticker, retries=3):
    """Tries to fetch data multiple times before giving up."""
    for i in range(retries):
        try:
            # We use download() as it's often more stable than Ticker.history()
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if not df.empty and len(df) > 100:
                return df
        except Exception:
            time.sleep(1) # Wait 1 second before retrying
    return None

def get_engine_data():
    results = []
    progress_bar = st.progress(0)
    
    for index, ticker in enumerate(watchlist):
        df = fetch_with_retry(ticker)
        
        if df is not None:
            try:
                t = yf.Ticker(ticker)
                info = t.info
                # Flatten the multi-index columns if they exist (common in yf.download)
                close_prices = df['Close']
                if isinstance(close_prices, pd.DataFrame):
                    curr_price = close_prices.iloc[-1].values[0]
                    series = close_prices.iloc[:, 0]
                else:
                    curr_price = close_prices.iloc[-1]
                    series = close_prices

                # --- TECHNICALS ---
                sma20 = series.rolling(window=20).mean().iloc[-1]
                sma50 = series.rolling(window=50).mean().iloc[-1]
                
                # AI Logic
                target = ((series.shift(-5) - series) / series) * 100
                X = series.tail(100).values.reshape(-1, 1) # Simplified features for stability
                y = target.tail(100).fillna(0).values
                
                model = RandomForestRegressor(n_estimators=30, random_state=42)
                model.fit(X[:-5], y[:-5])
                pred = model.predict(X[-1:])[0]

                results.append({
                    "Stock": ticker,
                    "Price": round(float(curr_price), 2),
                    "Signal %": round(float(pred), 2),
                    "ROE %": f"{round(info.get('returnOnEquity', 0) * 100, 1)}%",
                    "Sentiment": "BULLISH" if curr_price > sma20 else "BEARISH"
                })
            except:
                continue
        
        progress_bar.progress((index + 1) / len(watchlist))
    
    progress_bar.empty()
    return pd.DataFrame(results)

# --- 4. UI ---
st.title("💎 PRO AI Stock Engine")

with st.expander("🚀 How to bypass 'Fetch Failed'"):
    st.write("If you see an error, Yahoo is blocking the connection. Wait 30 seconds and click 'Run' again. The AI will try a different connection path.")

if st.button('Run AI Analysis 🚀'):
    with st.spinner('AI is analyzing market patterns...'):
        data = get_engine_data()
    
    if not data.empty:
        sorted_data = data.sort_values(by="Signal %", ascending=False)
        st.subheader(f"📈 Momentum Chart: {sorted_data.iloc[0]['Stock']}")
        chart_data = yf.download(sorted_data.iloc[0]['Stock'], period="6mo", progress=False)['Close']
        st.line_chart(chart_data)
        
        st.dataframe(sorted_data, use_container_width=True)
    else:
        st.error("Yahoo Finance is blocking requests. Try again in a minute.")
