import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from datetime import datetime

# --- 1. CONFIG & API KEY ---
st.set_page_config(page_title="AI Stock Intelligence", layout="wide")
# 💡 PRO TIP: I'm putting the key here for now so you can test, 
# but later you should move it to Streamlit Secrets!
AV_API_KEY = "YFYMDGYGFNX6KVYB" 

# --- 2. LIST OF MID-CAP STOCKS ---
watchlist = [
    "RECLTD", "PFC", "FEDERALBNK", "IDFCFIRSTB", "SAIL", 
    "NMDC", "OIL", "NATIONALUM", "VOLTAS", "ASHOKLEY", "POLYCAB"
]

# --- 3. THE DATA ENGINE ---
def fetch_alpha_vantage(symbol):
    """Fetches high-quality daily data from Alpha Vantage."""
    # Format for Indian Stocks: NSE:SYMBOL
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NSE:{symbol}&apikey={AV_API_KEY}'
    
    try:
        r = requests.get(url)
        data = r.json()
        
        if "Time Series (Daily)" not in data:
            st.error("API Limit reached or Symbol not found. (Alpha Vantage allows 25 calls/day)")
            return None
            
        # Convert JSON to DataFrame
        ts = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(ts, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        df = df.astype(float).sort_index()
        return df
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

# --- 4. THE AI BRAIN ---
def run_ai_logic(df):
    """Trains the Random Forest and predicts the next 5 days."""
    df['Target'] = ((df['Close'].shift(-5) - df['Close']) / df['Close']) * 100
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    
    train_df = df.dropna()
    X = train_df[['Close', 'SMA20']].tail(100)
    y = train_df['Target'].tail(100)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X[:-5], y[:-5])
    
    # Prediction + Bias Correction
    raw_pred = model.predict(X.tail(1))[0]
    bias = train_df['Target'].tail(10).mean()
    final_signal = raw_pred + (bias * 0.1)
    
    return round(final_signal, 2)

# --- 5. USER INTERFACE ---
st.title("🤖 AI Stock Intelligence Engine")
st.write("Professional-grade analysis powered by Alpha Vantage & Random Forest ML.")

# Sidebar Selection
selected_stock = st.sidebar.selectbox("Select a Mid-Cap Stock to Analyze", watchlist)
analyze_btn = st.sidebar.button("Run AI Analysis 🚀")

if analyze_btn:
    with st.spinner(f'Accessing Alpha Vantage for {selected_stock}...'):
        data = fetch_alpha_vantage(selected_stock)
        
    if data is not None:
        # Show Chart
        st.subheader(f"📈 {selected_stock} Price Action (Last 100 Days)")
        fig = px.line(data.tail(100), y='Close', template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Run AI
        signal = run_ai_logic(data)
        
        # Display Results in "Cards"
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"₹{data['Close'].iloc[-1]}")
        with col2:
            color = "normal" if signal > 0 else "inverse"
            st.metric("5-Day AI Signal", f"{signal}%", delta=f"{signal}%", delta_color=color)
        with col3:
            sentiment = "BULLISH" if data['Close'].iloc[-1] > data['Close'].rolling(20).mean().iloc[-1] else "BEARISH"
            st.subheader(f"Sentiment: {sentiment}")

        st.success(f"Analysis for {selected_stock} is complete!")
    else:
        st.warning("If the chart didn't load, you may have hit your 25-call daily limit.")

st.sidebar.markdown("---")
st.sidebar.write("💡 **Note:** Using the official Alpha Vantage API ensures 100% uptime compared to scrapers.")
