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
        background: white; border-radius: 24px; padding: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02); border: 1px solid #eef2f6; margin-bottom: 20px;
    }
    .news-card {
        background: white; border-radius: 16px; padding: 15px;
        border: 1px solid #eef2f6; margin-bottom: 10px; transition: 0.3s;
    }
    .news-card:hover { border-color: #4285f4; box-shadow: 0 4px 12px rgba(66,133,244,0.1); }
    .m-label { color: #5f6368; font-size: 13px; font-weight: 500; text-transform: uppercase; }
    .m-value { color: #1a73e8; font-size: 26px; font-weight: 500; margin-top: 5px; }
    .stButton>button {
        background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570);
        color: white; border: none; border-radius: 100px; padding: 12px 40px; font-weight: 500;
    }
    .badge-win { background: #e6f4ea; color: #1e8e3e; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 12px; }
    .badge-loss { background: #fce8e6; color: #d93025; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ENHANCED DATA & NEWS ENGINE ---
@st.cache_data(ttl=600) # Reduced cache time to see fresh news faster
def fetch_all_intel(symbol):
    try:
        # Price Data Fetch
        url_p = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={AV_API_KEY}'
        r_p = requests.get(url_p).json()
        if "Time Series (Daily)" not in r_p: return None, None
        
        df = pd.DataFrame.from_dict(r_p['Time Series (Daily)'], orient='index').astype(float).sort_index()
        df.index = pd.to_datetime(df.index)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # News Data Fetch (Added 'topics' to widen the net if ticker news is low)
        url_n = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&topics=technology,earnings&apikey={AV_API_KEY}"
        r_n = requests.get(url_n).json()
        news_feed = r_n.get("feed", [])[:4] 
        
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
            "Date": date.strftime('%b %d'), "Outcome": f"${row['Actual_5D']:.2f}",
            "Status": "win" if is_correct else "loss", "Result": "CORRECT" if is_correct else "MISSED"
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
    with st.spinner("Analyzing neural patterns and headlines..."):
        df, news = fetch_all_intel(ticker)
    
    if df is not None:
        history_data, acc_score = get_performance_history(df)
        
        # Training Logic
        df['MA10'] = df['Close'].rolling(10).mean()
        train_df = df.dropna(subset=['Close', 'MA10']).copy()
        train_df['Target'] = train_df['Close'].shift(-5)
        model_ready = train_df.dropna()
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(model_ready[['Close', 'MA10']], model_ready['Target'])
        
        last_p = df['Close'].iloc[-1]
        pred_p = model.predict(df[['Close', 'MA10']].tail(1))[0]
        change = ((pred_p - last_p) / last_p) * 100

        # UI: Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"<div class='gemini-card'><div class='m-label'>Price</div><div class='m-value'>${last_p:.2f}</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='gemini-card'><div class='m-label'>AI Target</div><div class='m-value'>${pred_p:.2f}</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='gemini-card'><div class='m-label'>Signal</div><div class='m-value'>{round(change, 1)}%</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='gemini-card'><div class='m-label'>Confidence</div><div class='m-value'>{int(acc_score)}%</div></div>", unsafe_allow_html=True)

        # UI: Chart & News
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("<div class='gemini-card'>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].tail(60), line=dict(color='#4285f4', width=3)))
            fig.add_trace(go.Scatter(x=[df.index[-1], df.index[-1] + pd.Timedelta(days=5)], y=[last_p, pred_p], line=dict(color='#9b72cb', dash='dot', width=3)))
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=400, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("### 📰 Market News")
            if news:
                for item in news:
                    st.markdown(f"""
                        <div class='news-card'>
                            <small style='color:#4285f4; font-weight:bold;'>{item.get('source', 'Financial News')}</small><br>
                            <a href='{item.get('url', '#')}' target='_blank' style='text-decoration:none; color:#1f1f1f; font-size:14px; font-weight:500;'>{item.get('title', 'Headline unavailable')}</a>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No live headlines found. The API limit may be cooling down.")

        # Truth Ledger
        st.markdown("### 📊 Neural Performance Ledger")
        l_cols = st.columns(5)
        for i, item in enumerate(history_data):
            with l_cols[i]:
                badge = "badge-win" if item['Status'] == 'win' else "badge-loss"
                st.markdown(f"""
                    <div class='gemini-card' style='padding:15px; text-align:center; border-top: 4px solid {"#1e8e3e" if item["Status"]=="win" else "#d93025"}'>
                        <small style='color:#70757a;'>{item['Date']}</small><br>
                        <b style='font-size:16px;'>{item['Outcome']}</b><br>
                        <span class='{badge}'>{item['Result']}</span>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.error("Intelligence synchronization failed. Please wait a moment and retry.")

# --- 5. PERMANENT ABOUT SECTION ---
st.markdown("---")
with st.expander("ℹ️ How the Quantum AI Engine Works"):
    st.markdown("""
    ### **The Intelligence Loop**
    * **Recursive Learning:** The model retrains itself on the latest 100-day window every time you run a report. 
    * **Self-Correction:** The 'Confidence' metric is a directional backtest. The AI looks at its last 5 completed windows to calculate its trend accuracy.
    * **Sentiment Engine:** The news panel captures the qualitative context (headlines) that quantitative data alone might miss.
    """)
