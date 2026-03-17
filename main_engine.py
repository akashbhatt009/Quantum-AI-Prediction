import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pytz
from datetime import datetime
import time

# --- 1. SETTINGS ---
st.set_page_config(page_title="PRO Institutional Stock Engine", layout="wide")
IST = pytz.timezone('Asia/Kolkata')

# --- 2. MID CAP WATCHLIST (High Turnover) ---
watchlist = [
    "FEDERALBNK.NS", "IDFCFIRSTB.NS", "L&TFH.NS", "GMRINFRA.NS", "RECLTD.NS", 
    "PFC.NS", "NATIONALUM.NS", "HINDCOPPER.NS", "SAIL.NS", "NMDC.NS", 
    "OIL.NS", "MANAPPURAM.NS", "ABCAPITAL.NS", "BANDHANBNK.NS", "CUB.NS", 
    "ESCORTS.NS", "GLENMARK.NS", "INDIACEM.NS", "TATACOMM.NS", "VOLTAS.NS", 
    "ASHOKLEY.NS", "CONCOR.NS", "PETRONET.NS", "TATACHEMICALS.NS", "MFSL.NS", 
    "SUNTV.NS", "RAMCOCEM.NS", "JUBLFOOD.NS", "AUBANK.NS", "POLYCAB.NS"
]

def get_engine_data():
    results = []
    progress_bar = st.progress(0)
    
    for index, ticker in enumerate(watchlist):
        try:
            time.sleep(0.2) # Prevent Yahoo Finance Blocking
            t = yf.Ticker(ticker)
            df = t.history(period="2y")
            
            if df.empty or len(df) < 100:
                continue
            
            info = t.info
            curr_price = df['Close'].iloc[-1]
            
            # --- TECHNICALS ---
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # --- AI PREDICTION & BIAS ---
            df['Target'] = ((df['Close'].shift(-5) - df['Close']) / df['Close']) * 100
            train_df = df.dropna()
            X = train_df[['Close', 'SMA20', 'SMA50']].tail(100)
            y = train_df['Target'].tail(100)
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X[:-5], y[:-5])
            
            raw_pred = model.predict(X.tail(1))[0]
            # Bias Correction: adjusting prediction based on recent model error
            bias_adjustment = train_df['Target'].tail(10).mean() 
            final_pred = raw_pred + (bias_adjustment * 0.1)

            # --- FINANCIAL PARAMETERS (ROE/ROCE) ---
            roe = info.get('returnOnEquity', 0) * 100
            roce = info.get('returnOnAssets', 0) * 1.5 * 100 # Proxy for ROCE if not direct
            mcap = info.get('marketCap', 0) / 10000000 # In Crores
            
            # Predictability Score (based on volatility)
            volatility = df['Close'].pct_change().std() * 100
            predictability = max(0, 100 - (volatility * 10))

            results.append({
                "Stock": ticker,
                "Price": round(curr_price, 2),
                "Signal %": round(final_pred, 2),
                "Predictability": f"{int(predictability)}%",
                "ROE %": f"{round(roe, 1)}%",
                "ROCE %": f"{round(roce, 1)}%",
                "Market Cap (Cr)": int(mcap),
                "Sentiment": "BULLISH" if curr_price > df['SMA20'].iloc[-1] else "BEARISH"
            })
        except Exception:
            continue
        
        progress_bar.progress((index + 1) / len(watchlist))
    
    progress_bar.empty()
    return pd.DataFrame(results)

# --- 3. STREAMLIT UI ---
st.title("💎 PRO Institutional Stock Engine")
st.write(f"Market Date: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} IST")

if st.button('Run Analysis 🚀'):
    data = get_engine_data()
    if not data.empty:
        # Highlight top signals
        st.dataframe(data.sort_values(by="Signal %", ascending=False), use_container_width=True)
        st.success(f"Analysis complete for {len(data)} Mid Cap stocks!")
    else:
        st.error("Could not fetch data. Please wait 1 minute and try again.")
