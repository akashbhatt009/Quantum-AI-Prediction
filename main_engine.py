import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# --- 1. GLOBAL PREMIUM STYLES ---
st.set_page_config(page_title="Quantum AI Terminal", layout="wide")

st.markdown("""
    <style>
    .main { background: #000000; }
    .stMetric {
        background: #111111 !important;
        border: 1px solid #222 !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }
    div[data-testid="stMetricValue"] { color: #00f2ff !important; font-family: 'Courier New'; font-weight: bold; }
    .news-card {
        background: #0a0a0a;
        border: 1px solid #1a1a1a;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 8px;
    }
    .watchlist-item {
        padding: 5px 10px;
        background: #111;
        border: 1px solid #333;
        border-radius: 5px;
        margin-bottom: 5px;
        color: #00f2ff;
        font-size: 14px;
        text-align: center;
    }
    h1, h2, h3 { color: white !important; font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

AV_API_KEY = "YFYMDGYGFNX6KVYB"

# --- 2. THE STABLE ENGINE ---
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

def run_backtest(df):
    success = 0
    test_days = 15
    temp = df.copy()
    temp['MA10'] = temp['Close'].rolling(10).mean()
    temp['Target'] = temp['Close'].shift(-5)
    valid_data = temp.dropna()
    if len(valid_data) < 20: return "N/A"
    test_set = valid_data.tail(test_days)
    for i in range(len(test_set)):
        row = test_set.iloc[i]
        if row['Target'] > row['Close']: success += 1
    return f"{int((success / test_days) * 100)}%"

# --- 3. WATCHLIST LOGIC ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["NVDA", "TSLA"]

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#00f2ff;'>⚡ QUANTUM</h2>", unsafe_allow_html=True)
    
    # Selection and Action
    ticker = st.selectbox("Asset Search", ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOGL", "AMZN", "META"])
    
    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("EXECUTE", use_container_width=True)
    with col2:
        if st.button("WATCH+", use_container_width=True):
            if ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(ticker)
    
    st.markdown("---")
    st.markdown("### 👁️ WATCHLIST")
    for item in st.session_state.watchlist:
        st.markdown(f"<div class='watchlist-item'>{item}</div>", unsafe_allow_html=True)
    
    if st.button("Clear Watchlist"):
        st.session_state.watchlist = []
        st.rerun()

# --- 5. MAIN DASHBOARD ---
if analyze_btn:
    with st.spinner("QUANTUM SYNC..."):
        data = fetch_data(ticker)
        news, mood = fetch_sentiment(ticker)

    if data is not None:
        df = data.copy()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['Target'] = df['Close'].shift(-5)
        train_df = df.dropna(subset=['Target', 'MA10'])
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train_df[['Close', 'MA10']], train_df['Target'])
        
        last_row = df[['Close', 'MA10']].tail(1)
        last_price = last_row['Close'].iloc[0]
        pred_price = model.predict(last_row)[0]
        
        pct = ((pred_price - last_price) / last_price) * 100
        accuracy_score = run_backtest(data)

        st.markdown(f"<h2>{ticker} // <span style='color:#555;'>MARKET_TERMINAL</span></h2>", unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LAST PRICE", f"${round(last_price, 2)}")
        m2.metric("MOOD", mood)
        m3.metric("AI TARGET", f"${round(pred_price, 2)}", delta=f"{round(pct, 2)}%")
        m4.metric("BACKTEST", accuracy_score)

        st.markdown("---")

        col_left, col_right = st.columns([2.5, 1])

        with col_left:
            last_date = data.index[-1]
            future_date = last_date + pd.Timedelta(days=5)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'].tail(90), line=dict(color='#00f2ff', width=2), name="History"))
            fig.add_trace(go.Scatter(x=[last_date, future_date], y=[last_price, pred_price], line=dict(color='#ff00ff', width=3, dash='dot'), marker=dict(size=10, symbol='diamond', color='#ff00ff'), name="Forecast"))
            fig.update_layout(template="plotly_dark", xaxis_showgrid=False, yaxis_showgrid=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### 📡 INTEL")
            if news:
                for a in news:
                    st.markdown(f"<div class='news-card'><small style='color:#ff00ff;'>{a['source']}</small><br><b style='color:white; font-size:14px;'>{a['title'][:60]}...</b><br><a href='{a['url']}' target='_blank' style='color:#00f2ff; text-decoration:none; font-size:12px;'>READ INTEL</a></div>", unsafe_allow_html=True)
            else:
                st.info("No news detected.")
    else:
        st.error("Engine Timeout. Try again in 60 seconds.")
