import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# --- 1. PREMIUM STYLING ---
st.set_page_config(page_title="Quantum AI Terminal", layout="wide")
st.markdown("""
    <style>
    .main { background: radial-gradient(circle at top right, #0f172a, #020617); }
    .stMetric { background: rgba(255, 255, 255, 0.03) !important; border: 1px solid rgba(0, 242, 255, 0.2) !important; border-radius: 12px !important; }
    div[data-testid="stMetricValue"] { color: #00f2ff !important; text-shadow: 0 0 10px rgba(0, 242, 255, 0.4); font-family: 'Courier New'; }
    .news-card { background: rgba(255, 255, 255, 0.02); border-left: 3px solid #ff00ff; padding: 10px; margin-bottom: 8px; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

AV_API_KEY = "YFYMDGYGFNX6KVYB"

# --- 2. THE ENGINE ---
@st.cache_data(ttl=3600)
def fetch_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={AV_API_KEY}'
    try:
        r = requests.get(url)
        data = r.json()
        if "Time Series (Daily)" not in data: return None
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index').astype(float).sort_index()
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
    except: return None

@st.cache_data(ttl=3600)
def fetch_sentiment(symbol):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={AV_API_KEY}"
    try:
        r = requests.get(url)
        data = r.json()
        feed = data.get("feed", [])
        if not feed: return None, "NEUTRAL"
        scores = [float(a['overall_sentiment_score']) for a in feed[:5]]
        avg = sum(scores)/len(scores)
        mood = "OPTIMISTIC" if avg > 0.1 else "NERVOUS" if avg < -0.1 else "NEUTRAL"
        return feed[:4], mood
    except: return None, "UNKNOWN"

# --- 3. DASHBOARD ---
with st.sidebar:
    st.title("⚡ QUANTUM")
    ticker = st.selectbox("Asset Search", ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOGL"])
    analyze_btn = st.button("RUN NEURAL ANALYSIS", use_container_width=True)

if analyze_btn:
    with st.spinner("SYNCING TEMPORAL DATA..."):
        data = fetch_data(ticker)
        news, mood = fetch_sentiment(ticker)

    if data is not None:
        # AI Logic
        df = data.copy()
        df['MA10'] = df['Close'].rolling(10).mean()
        train = df.dropna()
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train[['Close', 'MA10']], train['Close'].shift(-5).dropna())
        
        last_close = df['Close'].iloc[-1]
        pred_price = model.predict([[last_close, df['MA10'].iloc[-1]]])[0]
        accuracy = run_backtest(df) # <-- New Backtest Call

        # UI: HEADER
        st.markdown(f"<h1>{ticker} <span style='color:#444;'>//</span> AI TERMINAL</h1>", unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CURRENT", f"${round(last_close, 2)}")
        m2.metric("MOOD", mood)
        m3.metric("AI TARGET", f"${round(pred_price, 2)}", delta=f"{round(((pred_price-last_close)/last_close)*100, 2)}%")
        m4.metric("BACKTEST ACCURACY", accuracy)

        col_left, col_right = st.columns([2.5, 1])

        with col_left:
            # CHART
            last_date = data.index[-1]
            future_date = last_date + pd.Timedelta(days=5)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'].tail(90), fill='tozeroy', fillcolor='rgba(0, 242, 255, 0.05)', line=dict(color='#00f2ff', width=2), name="History"))
            fig.add_trace(go.Scatter(x=[last_date, future_date], y=[last_close, pred_price], line=dict(color='#ff00ff', width=3, dash='dot'), marker=dict(size=10, symbol='diamond', color='#ff00ff'), name="Forecast"))
            fig.update_layout(template="plotly_dark", xaxis_showgrid=False, yaxis_showgrid=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### 📡 INTEL_FEED")
            if news:
                for a in news:
                    st.markdown(f"<div class='news-card'><small style='color:#ff00ff;'>{a['source']}</small><br><b>{a['title'][:60]}...</b><br><a href='{a['url']}' style='color:#00f2ff; font-size:12px;'>VIEW</a></div>", unsafe_allow_html=True)
    else:
        st.error("LIMIT REACHED. WAIT 60 SECONDS.")

# Placeholder for run_backtest function (ensure it's defined at the top or bottom)
