import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# --- 1. GEMINI DESIGN SYSTEM ---
st.set_page_config(page_title="Quantum AI", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafd; }
    
    /* Gemini Soft Cards */
    .gemini-card {
        background: white;
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.1);
        border: 1px solid #eef2f6;
        margin-bottom: 20px;
    }
    
    /* Neural Insight Box */
    .insight-box {
        background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 100%);
        border-left: 5px solid #4285f4;
        border-radius: 16px;
        padding: 20px;
        margin-top: 20px;
        color: #1f1f1f;
        font-family: 'Google Sans', sans-serif;
    }

    /* Metric Styling */
    .metric-label { color: #5f6368; font-size: 13px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { color: #1a73e8; font-size: 28px; font-weight: 500; margin-top: 4px; }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570);
        color: white;
        border: none;
        border-radius: 100px;
        padding: 12px 30px;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

AV_API_KEY = "YFYMDGYGFNX6KVYB"

# --- 2. THE STABLE DATA ENGINE ---
@st.cache_data(ttl=3600)
def fetch_all_intel(symbol):
    try:
        # Price Data
        url_price = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={AV_API_KEY}'
        r_p = requests.get(url_price).json()
        df = pd.DataFrame.from_dict(r_p['Time Series (Daily)'], orient='index').astype(float).sort_index()
        df.index = pd.to_datetime(df.index)
        
        # News Sentiment
        url_news = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={AV_API_KEY}"
        r_n = requests.get(url_news).json()
        feed = r_n.get("feed", [])
        avg_sent = np.mean([float(a['overall_sentiment_score']) for a in feed[:5]]) if feed else 0
        
        return df, feed[:3], avg_sent
    except: return None, None, 0

# --- 3. UI LAYOUT ---
st.markdown("<h1 style='text-align: center; font-size: 42px; margin-bottom: 10px;'>Quantum Finance AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5f6368; margin-bottom: 40px;'>Select an asset to generate a neural trajectory and sentiment summary.</p>", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1, 1.5, 1])
with col_b:
    ticker = st.selectbox("", ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMD"], label_visibility="collapsed")
    analyze_btn = st.button("Generate Intelligence Report", use_container_width=True)

if analyze_btn:
    data, news, sent_score = fetch_all_intel(ticker)
    
    if data is not None:
        # AI Modeling
        df = data.copy()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['Target'] = df['Close'].shift(-5)
        train = df.dropna()
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train[['Close', 'MA10']], train['Target'])
        
        last_price = df['Close'].iloc[-1]
        last_date = df.index[-1]
        pred_price = model.predict(df[['Close', 'MA10']].tail(1))[0]
        future_date = last_date + pd.Timedelta(days=5)
        change_pct = ((pred_price - last_price) / last_price) * 100
        
        # --- TOP METRICS ---
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='gemini-card'><div class='metric-label'>Price</div><div class='metric-value'>${round(last_price, 2)}</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='gemini-card'><div class='metric-label'>AI Signal</div><div class='metric-value'>{'Bullish' if change_pct > 0 else 'Bearish'} ({round(change_pct, 1)}%)</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='gemini-card'><div class='metric-label'>Market Mood</div><div class='metric-value'>{'Positive' if sent_score > 0.1 else 'Cautionary'}</div></div>", unsafe_allow_html=True)

        # --- MAIN CHART ---
        st.markdown("<div class='gemini-card'>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].tail(60), line=dict(color='#4285f4', width=3), name="History"))
        fig.add_trace(go.Scatter(x=[last_date, future_date], y=[last_price, pred_price], line=dict(color='#9b72cb', width=3, dash='dot'), name="Neural Forecast"))
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=0,r=0,t=10,b=0), height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- NEURAL INSIGHT (THE "GEMINI" PART) ---
        sentiment_text = "optimistic" if sent_score > 0.1 else "neutral to bearish"
        direction_text = "upward momentum" if change_pct > 0 else "downward pressure"
        
        st.markdown(f"""
            <div class='insight-box'>
                <h3 style='margin-top:0; color:#4285f4;'>✨ Neural Insight</h3>
                Based on current market patterns and recent news, my analysis indicates <b>{direction_text}</b> for {ticker} over the next 5 days. 
                Recent headlines are currently <b>{sentiment_text}</b>, which aligns with the neural network's target of <b>${round(pred_price, 2)}</b>. 
                I recommend monitoring volume levels at the opening bell for confirmation.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("I couldn't reach the financial servers. Please try again in 60 seconds.")
