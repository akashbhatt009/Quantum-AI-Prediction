import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import pytz
from datetime import datetime

# --- 1. SETTINGS & WATCHLIST ---
st.set_page_config(page_title="Institutional Stock Engine", layout="wide")
IST = pytz.timezone('Asia/Kolkata')

watchlist = [
    "BEL.NS", "IRFC.NS", "NTPC.NS", "ITC.NS", "BANKBARODA.NS", 
    "ONGC.NS", "NHPC.NS", "TRIDENT.NS", "TATASTEEL.NS", "WIPRO.NS",
    "JIOFIN.NS", "SJVN.NS", "COALINDIA.NS", "HINDCOPPER.NS", "NATIONALUM.NS",
    "PNB.NS", "SOUTHBANK.NS", "IDFCFIRSTB.NS", "MINDACORP.NS", "20MICRONS.NS",
    "TATAPOWER.NS", "ZENTEC.NS", "OIL.NS", "RVNL.NS", "MOTHERSON.NS"
]

def get_engine_data():
    results = []
    for ticker in watchlist:
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="5y")
            if len(df) < 200: continue
            
            info = t.info
            curr_price = df['Close'].iloc[-1]
            
            # --- TECHNICALS ---
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = 100 - (100 / (1 + df['Close'].diff().gt(0).rolling(14).sum() / df['Close'].diff().lt(0).rolling(14).sum()))
            df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['Target'] = ((df['Close'].shift(-5) - df['Close']) / df['Close']) * 100
            
            # --- FUNDAMENTALS ---
            roce = info.get('returnOnAssets', 0) if info.get('returnOnAssets') else 0
            debt_to_equity = (info.get('debtToEquity', 0) / 100) if info.get('debtToEquity') else 0
            pe_ratio = info.get('trailingPE', 0) if info.get('trailingPE') else 0
            
            # --- AI BRAIN ---
            train_df = df.dropna()
            X_features = ['Close', 'RSI', 'Vol_Ratio', 'SMA20', 'SMA50']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(train_df[X_features][:-10], train_df['Target'][:-10])
            past_pred = model.predict(train_df[X_features].iloc[-10].values.reshape(1, -1))[0]
            bias = past_pred - train_df['Target'].iloc[-10]
            
            model.fit(train_df[X_features][:-5], train_df['Target'][:-5])
            raw_future = model.predict(train_df[X_features].tail(1))[0]
            corrected_pred = raw_future - bias

            sentiment = "BULLISH" if curr_price > df['SMA20'].iloc[-1] else "BEARISH"
            health_score = 100 - (min(pe_ratio, 30) + (debt_to_equity * 20)) + (min(roce * 100, 20))
            
            signal = "WAIT ⏳"
            if corrected_pred > 2.5 and health_score > 60 and curr_price > df['SMA50'].iloc[-1]:
                signal = "STRONG BUY 🚀"
            elif corrected_pred < -1.5 or health_score < 40:
                signal = "AVOID/EXIT ❌"

            results.append({
                "Stock": ticker,
                "Price": round(curr_price, 2),
                "RSI": int(df['RSI'].iloc[-1]),
                "D/E": round(debt_to_equity, 2),
                "ROCE %": f"{round(roce*100, 1)}%",
                "Pred %": round(corrected_pred, 2),
                "Health": int(health_score),
                "Sentiment": sentiment,
                "SIGNAL": signal
            })
        except: continue
    return pd.DataFrame(results)

# --- 2. STREAMLIT UI ---
st.title("💎 PRO Institutional Stock Engine")
st.markdown(f"**Market Date:** {datetime.now(IST).strftime('%B %d, %Y')} | **Time:** {datetime.now(IST).strftime('%H:%M:%S')} IST")

if st.button('Run Analysis 🚀'):
    with st.spinner('AI Engine is scanning the market...'):
        data = get_engine_data()
        if not data.empty:
            st.dataframe(data.sort_values(by="Pred %", ascending=False), use_container_width=True)
            st.success("Analysis Complete!")
        else:
            st.error("Could not fetch data. Please try again.")
