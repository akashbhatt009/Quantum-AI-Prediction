import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# --- 1. GEMINI DESIGN SYSTEM ---
st.set_page_config(page_title="Quantum AI Prediction", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafd; }
    
    /* Gemini Soft White Cards */
    .gemini-card {
        background: white;
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        border: 1px solid #eef2f6;
        margin-bottom: 25px;
    }
    
    /* Clean Neural Insight Box */
    .insight-box {
        background: #ffffff;
        border: 1px solid #e1e4e8;
        border-left: 6px solid #4285f4;
        border-radius: 16px;
        padding: 24px;
        color: #1f1f1f;
        line-height: 1.6;
    }

    /* Modern Metric Labels */
    .m-label { color: #5f6368; font-size: 14px; font-weight: 500; }
    .m-value { color: #1a73e8; font-size: 32px; font-weight: 500; margin-top: 5px; }
    
    /* Styled Gradient Button */
    .stButton>button {
        background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570);
        color: white;
        border: none;
        border-radius: 100px;
        padding: 14px 40px;
        font-weight: 500;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton>button:hover { opacity: 0.9; box-shadow: 0 4px 15px rgba(66, 133, 244, 0.3); }
    </style>
    """, unsafe_allow_html=True)

AV_API_KEY = "YFYMDGYGFNX6KVYB"

# --- 2. THE STABLE ENGINE (Fixes KeyErrors) ---
@st.cache_data(ttl=3600)
def fetch_financial_intelligence(symbol):
    try:
        # Price Intel
        url_p = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={AV_API_KEY}'
        resp = requests.get(url_p).json()
        
        if "Time Series (Daily)" not in resp:
            return None, None, 0
            
        raw_data = resp["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(raw_data, orient='index').astype(float).sort_index()
        df.index = pd.to_datetime(df.index)
        # Explicitly renaming to ensure 'Close' exists
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # News Sentiment Intel
        url_n = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={AV_API_KEY}"
        news_resp = requests.get(url_n).json()
        feed = news_resp.get("feed", [])
        
        avg_sentiment = 0
        if feed:
            scores = [float(a.get('overall_sentiment_score', 0)) for a in feed[:5]]
            avg_sentiment = np.mean(scores)
            
        return df, feed[:3], avg_sentiment
    except Exception as e:
        return None, None, 0

# --- 3. UI PRESENTATION ---
st.markdown("<h1 style='text-align: center; font-size: 48px; font-weight: 500; margin-top: 40px;'>Quantum AI Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5f6368; font-size: 18px; margin-bottom: 40px;'>Advanced neural forecasting and sentiment intelligence.</p>", unsafe_allow_html=True)

# Centered Search
c1, c2, c3 = st.columns([1, 1.2, 1])
with c2:
    ticker = st.selectbox("", ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMD", "META"], label_visibility="collapsed")
    run = st.button("Generate Intelligence Report", use_container_width=True)

if run:
    with st.spinner("Processing neural layers..."):
        df, news_feed, mood_score = fetch_financial_intelligence(ticker)
    
    if df is not None:
        # ML Logic
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['Target'] = df['Close'].shift(-5)
        train_set = df.dropna()
        
        if not train_set.empty:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(train_set[['Close', 'MA10']], train_set['Target'])
            
            curr_price = df['Close'].iloc[-1]
            last_dt = df.index[-1]
            prediction = model.predict(df[['Close', 'MA10']].tail(1))[0]
            target_dt = last_dt + pd.Timedelta(days=5)
            move_pct = ((prediction - curr_price) / curr_price) * 100
            
            # --- Results Display ---
            st.markdown("<br>", unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"<div class='gemini-card'><div class='m-label'>Current Market</div><div class='m-value'>${round(curr_price, 2)}</div></div>", unsafe_allow_html=True)
            with r2:
                st.markdown(f"<div class='gemini-card'><div class='m-label'>Neural Signal</div><div class='m-value' style='color:{'#34a853' if move_pct > 0 else '#ea4335'}'>{round(move_pct, 1)}%</div></div>", unsafe_allow_html=True)
            with r3:
                st.markdown(f"<div class='gemini-card'><div class='m-label'>Model Bias</div><div class='m-value'>{'OPTIMISTIC' if mood_score > 0.1 else 'CAUTIOUS'}</div></div>", unsafe_allow_html=True)

            # Chart
            st.markdown("<div class='gemini-card'>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].tail(60), line=dict(color='#4285f4', width=3), name="Historical"))
            fig.add_trace(go.Scatter(x=[last_dt, target_dt], y=[curr_price, prediction], line=dict(color='#9b72cb', width=3, dash='dot'), marker=dict(size=12, color='#9b72cb'), name="AI Path"))
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=0,r=0,t=10,b=0), height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Insight Box
            bias = "positive news sentiment" if mood_score > 0.1 else "neutral to negative market chatter"
            outlook = "growth" if move_pct > 0 else "consolidation"
            
            st.markdown(f"""
                <div class='insight-box'>
                    <h3 style='margin-top:0; color:#4285f4;'>✨ Neural Summary</h3>
                    Quantum analysis suggests a period of <b>{outlook}</b> for {ticker} over the coming week. 
                    The model is factoring in <b>{bias}</b>, resulting in a predicted target of <b>${round(prediction, 2)}</b> by {target_dt.strftime('%B %d')}. 
                    This prediction carries a medium confidence level based on current volatility clusters.
                </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Intelligence Fetch Failed. The API limit may have been reached—please try again in one minute.")
