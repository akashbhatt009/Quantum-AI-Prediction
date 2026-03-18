import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from datetime import datetime

# --- 1. GEMINI AI STYLE CONFIG ---
st.set_page_config(page_title="Quantum AI", layout="wide")

st.markdown("""
    <style>
    /* Gemini-style soft background */
    .main { background-color: #f8fafd; }
    
    /* Clean, soft cards instead of black boxes */
    .gemini-card {
        background: white;
        border-radius: 24px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        border: 1px solid #eef2f6;
        margin-bottom: 20px;
    }
    
    /* Typography */
    h1, h2, h3 { color: #1f1f1f; font-family: 'Google Sans', sans-serif; font-weight: 400; }
    .metric-label { color: #70757a; font-size: 14px; margin-bottom: 8px; }
    .metric-value { color: #1a73e8; font-size: 32px; font-weight: 500; }
    
    /* Gradient Button */
    .stButton>button {
        background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 10px 25px;
        font-weight: 500;
        transition: 0.3s;
    }
    .stButton>button:hover { opacity: 0.9; transform: scale(1.02); }
    </style>
    """, unsafe_allow_html=True)

AV_API_KEY = "YFYMDGYGFNX6KVYB"

# --- 2. ROBUST ENGINE (Fixes TypeError) ---
@st.cache_data(ttl=3600)
def fetch_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={AV_API_KEY}'
    try:
        r = requests.get(url)
        data = r.json()
        if "Time Series (Daily)" not in data: return None
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index').astype(float).sort_index()
        # FIX: Convert index to datetime objects explicitly
        df.index = pd.to_datetime(df.index)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
    except: return None

# --- 3. UI LAYOUT ---
st.markdown("<h1 style='text-align: center; margin-top: 50px;'>How can I help you with the markets today?</h1>", unsafe_allow_html=True)

# Centered Search Bar
col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b:
    ticker = st.selectbox("", ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL"], label_visibility="collapsed")
    analyze_btn = st.button("Generate Neural Insights", use_container_width=True)

if analyze_btn:
    data = fetch_data(ticker)
    
    if data is not None:
        # AI Processing
        df = data.copy()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['Target'] = df['Close'].shift(-5)
        train_df = df.dropna()
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train_df[['Close', 'MA10']], train_df['Target'])
        
        last_row = df[['Close', 'MA10']].tail(1)
        last_price = last_row['Close'].iloc[0]
        # FIX: last_date is now a proper Timestamp object
        last_date = df.index[-1]
        future_date = last_date + pd.Timedelta(days=5) 
        
        pred_price = model.predict(last_row)[0]
        change = ((pred_price - last_price) / last_price) * 100

        # --- GEMINI RESULTS UI ---
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metric Row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='gemini-card'><p class='metric-label'>Current Price</p><p class='metric-value'>${round(last_price, 2)}</p></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='gemini-card'><p class='metric-label'>AI Prediction</p><p class='metric-value' style='color:#34a853;'>${round(pred_price, 2)}</p></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='gemini-card'><p class='metric-label'>5-Day Forecast</p><p class='metric-value'>{round(change, 2)}%</p></div>", unsafe_allow_html=True)

        # Visualization
        st.markdown("<div class='gemini-card'>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].tail(60), line=dict(color='#4285f4', width=3), name="History"))
        fig.add_trace(go.Scatter(x=[last_date, future_date], y=[last_price, pred_price], line=dict(color='#9b72cb', width=3, dash='dot'), marker=dict(size=12, symbol='circle', color='#9b72cb'), name="AI Path"))
        
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=False, color='#70757a'),
            yaxis=dict(showgrid=True, gridcolor='#f1f3f4', color='#70757a'),
            margin=dict(l=0, r=0, t=20, b=0), height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.error("I'm having trouble retrieving that data. Please try again in a moment.")
