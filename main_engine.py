import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# --- 1. TERMINAL STYLING (The "Sophistication" Layer) ---
st.set_page_config(page_title="AI Market Terminal", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #333; padding: 15px; border-radius: 10px; }
    div[data-testid="stMetricValue"] { color: #00d4ff; font-family: 'Courier New', Courier, monospace; }
    .news-card { border: 1px solid #333; padding: 12px; border-radius: 8px; margin-bottom: 10px; background-color: #1a1c24; transition: 0.3s; }
    .news-card:hover { border-color: #00d4ff; background-color: #252936; }
    </style>
    """, unsafe_allow_html=True)

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
        df = pd.DataFrame.from_dict(ts, orient='index').astype(float).sort_index()
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df.tail(120)
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
        mood = "Optimistic 🚀" if avg_score > 0.15 else "Nervous 📉" if avg_score < -0.15 else "Neutral ⚖️"
        return feed[:5], mood
    except: return None, "Unknown"

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("🎮 Command Center")
ticker = st.sidebar.selectbox("Select Asset", ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOGL", "AMZN"])
analyze_btn = st.sidebar.button("Execute AI Analysis 🚀")

# --- 4. MAIN DASHBOARD ---
if analyze_btn:
    with st.spinner(f"Running Quantum Models on {ticker}..."):
        data = fetch_us_stock_data(ticker)
        news, mood = fetch_sentiment(ticker)
    
    if data is not None:
        # AI Calculations
        df = data.copy()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['Target'] = df['Close'].shift(-5)
        train = df.dropna()
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train[['Close', 'MA10']], train['Target'])
        
        pred_price = model.predict(df[['Close', 'MA10']].tail(1).values)[0]
        pct_change = ((pred_price - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
        confidence = max(0, 100 - int(df['Close'].pct_change().std() * 1500))

        # UI: HEADER SECTION
        st.markdown(f"<h1 style='color: #00d4ff;'>{ticker} | AI TERMINAL</h1>", unsafe_allow_html=True)
        
        # UI: TOP METRIC ROW
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Price", f"${round(df['Close'].iloc[-1], 2)}")
        m2.metric("Market Mood", mood)
        m3.metric("AI Signal", f"{round(pct_change, 2)}%", delta=f"{round(pct_change, 1)}%")
        m4.metric("Confidence", f"{confidence}%")

        st.markdown("---")

        # UI: TWO-COLUMN BODY
        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.subheader("📊 Price Momentum Analysis")
            fig = px.line(data.tail(90), y='Close', template="plotly_dark", color_discrete_sequence=['#00d4ff'])
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical Pulse Sub-Row
            st.subheader("🛠️ Technical Pulse")
            p1, p2, p3 = st.columns(3)
            p1.write(f"**90D High:** ${round(data['High'].max(), 2)}")
            p2.write(f"**90D Low:** ${round(data['Low'].min(), 2)}")
            p3.write(f"**Volume Status:** {'High' if df['Volume'].iloc[-1] > df['Volume'].mean() else 'Normal'}")

        with right_col:
            st.subheader("📰 Intelligence Feed")
            if news:
                for article in news:
                    st.markdown(f"""
                    <div class="news-card">
                        <p style="font-size: 14px; margin-bottom: 5px; color: #fff;"><b>{article['title'][:70]}...</b></p>
                        <p style="font-size: 11px; color: #888; margin-bottom: 5px;">Source: {article['source']} | Mood: {article['overall_sentiment_label']}</p>
                        <a href="{article['url']}" style="color: #00d4ff; text-decoration: none; font-size: 12px;">Read Full Intel →</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No fresh news available. AI is operating on pure price action.")

        st.success(f"Full analysis for {ticker} complete.")
    else:
        st.error("API Limit Reached (25/day). Check back tomorrow or refresh with a new key.")
else:
    st.info("👈 Select a stock from the Command Center to begin.")
