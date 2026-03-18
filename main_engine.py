import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# --- 1. PREMIUM TERMINAL STYLING ---
st.set_page_config(page_title="Quantum AI Terminal", layout="wide")

st.markdown("""
    <style>
    .main { background: radial-gradient(circle at top right, #0f172a, #020617); }
    .stMetric {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(0, 242, 255, 0.2) !important;
        border-radius: 15px !important;
        padding: 15px !important;
    }
    div[data-testid="stMetricValue"] { color: #00f2ff !important; text-shadow: 0 0 10px rgba(0, 242, 255, 0.4); font-family: 'Courier New'; }
    .news-card {
        background: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #ff00ff;
        padding: 12px;
        margin-bottom: 10px;
        border-radius: 4px;
        transition: 0.3s;
    }
    .news-card:hover { background: rgba(255, 255, 255, 0.05); transform: translateX(5px); }
    </style>
    """, unsafe_allow_html=True)

AV_API_KEY = "YFYMDGYGFNX6KVYB"

# --- 2. THE ENGINE FUNCTIONS ---
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
        mood = "OPTIMISTIC 🚀" if avg > 0.1 else "NERVOUS 📉" if avg < -0.1 else "NEUTRAL ⚖️"
        return feed[:4], mood
    except: return None, "UNKNOWN"

def run_backtest(df):
    """Calculates directional accuracy over last 15 valid windows."""
    success = 0
    test_range = 15 
    temp_df = df.copy()
    temp_df['MA10'] = temp_df['Close'].rolling(10).mean()
    
    # We loop backwards to check past predictions vs what actually happened
    for i in range(test_range + 5, 5, -1):
        train = temp_df.iloc[:-i].dropna()
        if len(train) < 20: continue
        
        target = temp_df['Close'].shift(-5).iloc[:-i].dropna()
        # Align X and Y
        X_train = train.loc[target.index, ['Close', 'MA10']]
        
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        model.fit(X_train, target)
        
        pred = model.predict(train[['Close', 'MA10']].tail(1))[0]
        actual = temp_df['Close'].iloc[-i+5]
        
        # Check if direction matched
        if (pred > train['Close'].iloc[-1] and actual > train['Close'].iloc[-1]) or \
           (pred < train['Close'].iloc[-1] and actual < train['Close'].iloc[-1]):
            success += 1
            
    return f"{int((success/test_range)*100)}%"

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("⚡ QUANTUM")
    ticker = st.selectbox("Asset Search", ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOGL", "AMZN"])
    analyze_btn = st.button("RUN NEURAL ANALYSIS", use_container_width=True)
    st.markdown("---")
    st.caption("Status: API Active (Free Tier)")

# --- 4. MAIN INTERFACE ---
if analyze_btn:
    with st.spinner("SYNCING TEMPORAL DATA..."):
        data = fetch_data(ticker)
        news, mood = fetch_sentiment(ticker)

    if data is not None:
        # AI LOGIC: PREVENTING VALUEERROR
        df = data.copy()
        df['MA10'] = df['Close'].rolling(10).mean()
        
        # Prepare Training Sets (Shift targets but drop NaNs to align lengths)
        df['Target'] = df['Close'].shift(-5)
        train_df = df.dropna(subset=['Target', 'MA10'])
        
        X_train = train_df[['Close', 'MA10']]
        y_train = train_df['Target']
        
        # Train Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict using the very latest row (Today)
        latest_row = df[['Close', 'MA10']].tail(1)
        last_close = latest_row['Close'].iloc[0]
        pred_price = model.predict(latest_row)[0]
        
        # Metrics
        pct_change = ((pred_price - last_close) / last_close) * 100
        accuracy = run_backtest(data)
        confidence = max(0, 100 - int(data['Close'].pct_change().std() * 1800))

        # UI: HEADER
        st.markdown(f"<h1>{ticker} <span style='color:#444;'>//</span> SYSTEM DASHBOARD</h1>", unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CURRENT", f"${round(last_close, 2)}")
        m2.metric("MOOD", mood)
        m3.metric("AI TARGET (5D)", f"${round(pred_price, 2)}", delta=f"{round(pct_change, 2)}%")
        m4.metric("BACKTEST ACCURACY", accuracy)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([2.5, 1])

        with col_left:
            # DYNAMIC CHART
            last_date = data.index[-1]
            future_date = last_date + pd.Timedelta(days=5)
            
            fig = go.Figure()
            # Historical
            fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'].tail(90), fill='tozeroy', fillcolor='rgba(0, 242, 255, 0.05)', line=dict(color='#00f2ff', width=2), name="History"))
            # Prediction
            fig.add_trace(go.Scatter(x=[last_date, future_date], y=[last_close, pred_price], line=dict(color='#ff00ff', width=3, dash='dot'), marker=dict(size=10, symbol='diamond', color='#ff00ff'), name="AI Forecast"))
            
            fig.update_layout(template="plotly_dark", xaxis_showgrid=False, yaxis_showgrid=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=480, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### 📡 INTEL_FEED")
            if news:
                for a in news:
                    st.markdown(f"""
                    <div class='news-card'>
                        <small style='color:#ff00ff;'>{a['source']} • {a['overall_sentiment_label']}</small><br>
                        <p style='margin:5px 0; font-size:14px;'><b>{a['title'][:65]}...</b></p>
                        <a href='{a['url']}' style='color:#00f2ff; font-size:12px; text-decoration:none;'>VIEW INTEL →</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No fresh news intel detected for this asset.")
    else:
        st.error("API Error: Please check if the symbol is correct or if the 25-call daily limit was reached.")
else:
    st.markdown("<div style='text-align:center; margin-top:100px;'><h2 style='color:#333;'>SYSTEM STANDBY</h2><p>Initiate analysis from the command center.</p></div>", unsafe_allow_html=True)
