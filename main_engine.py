import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# --- 1. CONFIG ---
st.set_page_config(page_title="AI Global Stock Analyst", layout="wide")
AV_API_KEY = "YFYMDGYGFNX6KVYB" 

# --- 2. DATA ENGINES ---
@st.cache_data(ttl=3600)
def fetch_us_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={AV_API_KEY}'
    try:
        r = requests.get(url)
        data = r.json()
        if "Time Series (Daily)" not in data: return None
        ts = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(ts, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        return df.astype(float).sort_index().tail(120)
    except: return None

@st.cache_data(ttl=3600)
def fetch_sentiment(symbol):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={AV_API_KEY}"
    try:
        r = requests.get(url)
        data = r.json()
        feed = data.get("feed", [])
        if not feed: return None, "Neutral"
        scores = [float(article['overall_sentiment_score']) for article in feed[:5]]
        avg_score = sum(scores) / len(scores)
        if avg_score > 0.15: mood = "Optimistic 🚀"
        elif avg_score < -0.15: mood = "Nervous 📉"
        else: mood = "Neutral ⚖️"
        return feed[:3], mood
    except: return None, "Unknown"

# --- 3. UI LAYOUT ---
st.title("🌎 AI Global Market Intelligence")
st.markdown("---")

ticker = st.sidebar.selectbox("Select Stock:", ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOGL"])
analyze_btn = st.sidebar.button("Generate AI Forecast 🚀")

if analyze_btn:
    with st.spinner(f"Analyzing {ticker}..."):
        data = fetch_us_stock_data(ticker)
        news, mood = fetch_sentiment(ticker)
    
    if data is not None:
        # --- AI CALCULATIONS ---
        df = data.copy()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['Target'] = df['Close'].shift(-5)
        train = df.dropna()
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train[['Close', 'MA10']], train['Target'])
        
        pred_price = model.predict(df[['Close', 'MA10']].tail(1).values)[0]
        pct_change = ((pred_price - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
        confidence = max(0, 100 - int(df['Close'].pct_change().std() * 1500))

        # --- DISPLAY METRICS ---
        st.subheader(f"🤖 AI Analysis for {ticker}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Market Mood", mood)
        c2.metric("AI 5-Day Signal", f"{round(pct_change, 2)}%", delta=f"{round(pct_change, 1)}%")
        c3.metric("Model Confidence", f"{confidence}%")

        # --- CHART ---
        fig = px.line(data.tail(90), y='Close', template="plotly_dark", color_discrete_sequence=['#00D4FF'])
        st.plotly_chart(fig, use_container_width=True)

        # --- NEWS TICKER ---
        st.markdown("---")
        st.subheader("📰 Recent Intelligence")
        if news:
            for article in news:
                with st.expander(f"NEWS: {article['title']}"):
                    st.write(f"**Source:** {article['source']} | **Mood:** {article['overall_sentiment_label']}")
                    st.write(f"[Read Full Story]({article['url']})")
    else:
        st.error("API Limit Reached or Data Unavailable. Try again later.")
