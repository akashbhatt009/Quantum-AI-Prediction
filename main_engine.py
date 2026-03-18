import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# --- 1. CONFIG ---
st.set_page_config(page_title="AI Global Stock Analyst", layout="wide")
AV_API_KEY = "YFYMDGYGFNX6KVYB" 

# --- 2. THE DATA ENGINE (US Optimized) ---
@st.cache_data(ttl=3600)
def fetch_us_stock_data(symbol):
    # US stocks don't need a prefix like 'NSE:' or 'BSE:'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={AV_API_KEY}'
    
    try:
        r = requests.get(url)
        data = r.json()
        
        if "Note" in data:
            st.error("⏳ API Limit Reached (25 calls/day). Please try again in 24 hours.")
            return None
            
        if "Time Series (Daily)" not in data:
            st.warning(f"Could not find data for {symbol}. Ensure the ticker is correct.")
            return None
            
        ts = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(ts, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        return df.astype(float).sort_index().tail(120) # 120 days for training
    except:
        return None

# --- 3. UI ---
st.title("🌎 AI Global Market Intelligence")
st.markdown("---")

# High-Liquidity US Watchlist
ticker = st.selectbox("Select a High-Growth US Stock:", 
                      ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOGL", "AMZN", "META"])

if st.button("Generate AI Forecast 🚀"):
    with st.spinner(f"Analyzing {ticker} patterns..."):
        data = fetch_us_stock_data(ticker)
    
    if data is not None:
        # 📈 Visual Trend
        st.subheader(f"90-Day Price Trend: {ticker}")
        fig = px.line(data.tail(90), y='Close', template="plotly_dark", 
                     color_discrete_sequence=['#00D4FF'])
        st.plotly_chart(fig, use_container_width=True)
        
        # 🧠 Machine Learning (Random Forest)
        df = data.copy()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['Target'] = df['Close'].shift(-5) # Predicting 5 days out
        
        train = df.dropna()
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train[['Close', 'MA10']], train['Target'])
        
        # Latest Prediction
        last_val = df[['Close', 'MA10']].tail(1).values
        pred_price = model.predict(last_val)[0]
        pct_change = ((pred_price - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
        
        # 📊 Professional Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Live Price", f"${round(df['Close'].iloc[-1], 2)}")
        
        # Color logic for prediction
        color = "normal" if pct_change > 0 else "inverse"
        col2.metric("AI 5-Day Signal", f"{round(pct_change, 2)}%", delta=f"{round(pct_change, 1)}%", delta_color=color)
        
        # Bias/Predictability logic
        volatility = df['Close'].pct_change().std() * 100
        confidence = max(0, 100 - int(volatility * 15))
        col3.metric("Model Confidence", f"{confidence}%")

        st.info(f"**Insight:** The model is currently {('BULLISH' if pct_change > 0 else 'BEARISH')} on {ticker} based on {len(train)} training iterations.")
