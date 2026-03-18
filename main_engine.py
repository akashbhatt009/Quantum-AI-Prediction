import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# --- 1. SECURE API CONFIGURATION ---
try:
    AV_API_KEY = st.secrets["AV_API_KEY"]
except:
    st.error("API Key not found. Please configure AV_API_KEY in Streamlit Secrets.")
    st.stop()

# --- 2. GEMINI DESIGN SYSTEM ---
st.set_page_config(page_title="Quantum AI Prediction", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafd; }
    .gemini-card {
        background: white;
        border-radius: 24px;
        padding: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        border: 1px solid #eef2f6;
        margin-bottom: 20px;
    }
    .news-card {
        background: white;
        border-radius: 16px;
        padding: 15px;
        border: 1px solid #eef2f6;
        margin-bottom: 10px;
        transition: 0.3s;
    }
    .news-card:hover { border-color: #4285f4; box-shadow: 0 4px 12px rgba(66,133,244,0.1); }
    .insight-box {
        background: #ffffff;
        border-left: 6px solid #4285f4;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 25px;
        border: 1px solid #e1e4e8;
    }
    .m-label { color: #5f6368; font-size: 13px; font-weight: 500; text-transform: uppercase; }
    .m-value { color: #1a73e8; font-size: 26px; font-weight: 500; margin-top: 5px; }
    .stButton>button {
        background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570);
        color: white; border: none; border-radius: 100px; padding: 12px 40px;
        font-weight: 500;
    }
    .badge-win { background: #e6f4ea; color: #1e8e3e; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 12px; }
    .badge-loss { background: #fce8e6; color: #d93025; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA & NEWS ENGINE ---
@st.cache_data(ttl=3600)
def fetch_all_intel(symbol):
    try:
        # Price Data
        url_p = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={AV_API_KEY}'
        r_p = requests.get(url_p).json()
        if "Time Series (Daily)" not in r_p: return None, None
        df = pd.DataFrame.from_dict(r_p['Time Series (Daily)'], orient='index').astype(float).sort_index()
        df.index = pd.to_datetime(df.index)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # News Data
        url_n = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={AV_API_KEY}"
        r_n = requests.get(url_n).json()
        news_feed = r_n.get("feed", [])[:3] # Get top 3 headlines
        
        return df, news_feed
    except: return None, None

def get_performance_history(df):
    temp = df.copy()
    temp['MA10'] = temp['Close'].rolling(10).mean()
    temp['Actual_5D'] = temp['Close'].shift(-5)
    valid = temp.dropna().tail(5)
    history = []
    correct_count = 0
    for date, row in valid.iterrows():
        predicted_up = row['MA10'] > row['Close']
        actual_up = row['Actual_5D'] > row['Close']
        is_correct = (predicted_up == actual_up)
        if is_correct: correct_count += 1
        history.append({
            "Date": date.strftime('%b %d'),
            "Outcome": f"${row['Actual_5D']:.2f}",
            "Status": "win" if is_correct else "loss",
            "Result": "CORRECT" if is_correct else "MISSED"
        })
    accuracy = (correct_count / 5) * 100 if len(valid) > 0 else 0
    return history, accuracy

# --- 4. INTERFACE ---
st.markdown("<h1 style='text-align: center; margin-top: 30px;'>Quantum AI Prediction</h1>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1.2, 1])
with c2:
    ticker = st.selectbox("", ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMD"], label_visibility="collapsed")
    run = st.button("Generate Intelligence Report", use_container_width=True)

if run:
    df, news = fetch_all_intel(ticker)
    if df is not None:
        history_data, acc_score = get_performance_history(df)
        
        # ML Training
        df['MA10'] = df['Close'].rolling(10).mean()
        train_df = df.dropna(subset=['Close', 'MA10']).copy()
        train_df['Target'] = train
