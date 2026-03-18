import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# --- 1. CONFIG ---
st.set_page_config(page_title="AI Stock Analyst", layout="wide")
AV_API_KEY = "YFYMDGYGFNX6KVYB" 

# --- 2. THE SMART FETCH (With Caching) ---
@st.cache_data(ttl=3600) # Caches data for 1 hour to save API calls
def fetch_stock_data(symbol):
    # Alpha Vantage NSE format: NSE:SYMBOL
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NSE:{symbol}&apikey={AV_API_KEY}'
    
    try:
        r = requests.get(url)
        data = r.json()
        
        # Check if we hit the limit
        if "Note" in data or "Information" in data:
            st.error("🚫 Alpha Vantage Daily Limit (25) reached. Try again tomorrow or use a new key.")
            return None
            
        if "Time Series (Daily)" not in data:
            st.warning(f"No data found for {symbol}. checking symbol format...")
            return None
            
        ts = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(ts, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        return df.astype(float).sort_index().tail(100)
    except:
        return None

# --- 3. UI ---
st.title("🤖 AI Deep-Dive Analyst")
st.markdown(f"**API Status:** Free Tier (Used for professional NSE data)")

# Use a smaller, focused list
ticker = st.selectbox("Pick a stock to analyze:", ["RECLTD", "FEDERALBNK", "PFC", "SAIL", "NMDC", "VOLTAS"])

if st.button("Start AI Analysis"):
    data = fetch_stock_data(ticker)
    
    if data is not None:
        # 📈 Visuals
        st.subheader(f"Momentum: {ticker}")
        fig = px.line(data, y='Close', template="plotly_dark", color_discrete_sequence=['#00ffcc'])
        st.plotly_chart(fig, use_container_width=True)
        
        # 🧠 Simple AI Logic (Random Forest)
        df = data.copy()
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['Target'] = df['Close'].shift(-5)
        
        train = df.dropna()
        model = RandomForestRegressor(n_estimators=50)
        model.fit(train[['Close', 'SMA10']], train['Target'])
        
        last_features = df[['Close', 'SMA10']].tail(1).values
        pred_price = model.predict(last_features)[0]
        change = ((pred_price - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
        
        # 📊 Results
        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"₹{round(df['Close'].iloc[-1], 2)}")
        c2.metric("AI 5-Day Prediction", f"{round(change, 2)}%", delta=f"{round(change, 1)}%")
