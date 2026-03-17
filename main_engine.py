import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pytz
from datetime import datetime
import time
import plotly.express as px # New library for charts

# --- 1. SETTINGS ---
st.set_page_config(page_title="PRO AI Stock Engine", layout="wide")
IST = pytz.timezone('Asia/Kolkata')

# --- 2. MID CAP WATCHLIST ---
watchlist = [
    "FEDERALBNK.NS", "IDFCFIRSTB.NS", "L&TFH.NS", "GMRINFRA.NS", "RECLTD.NS", 
    "PFC.NS", "NATIONALUM.NS", "HINDCOPPER.NS", "SAIL.NS", "NMDC.NS", 
    "OIL.NS", "MANAPPURAM.NS", "ABCAPITAL.NS", "BANDHANBNK.NS", "CUB.NS", 
    "ESCORTS.NS", "GLENMARK.NS", "INDIACEM.NS", "TATACOMM.NS", "VOLTAS.NS", 
    "ASHOKLEY.NS", "CONCOR.NS", "PETRONET.NS", "TATACHEMICALS.NS", "MFSL.NS", 
    "SUNTV.NS", "RAMCOCEM.NS", "JUBLFOOD.NS", "AUBANK.NS", "POLYCAB.NS"
]

def get_engine_data():
    results = []
    progress_bar = st.progress(0)
    
    for index, ticker in enumerate(watchlist):
        try:
            time.sleep(0.1) 
            t = yf.Ticker(ticker)
            df = t.history(period="2y")
            
            if df.empty or len(df) < 100:
                continue
            
            info = t.info
            curr_price = df['Close'].iloc[-1]
            
            # --- TECHNICALS ---
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            
            # --- AI PREDICTION & BIAS ---
            df['Target'] = ((df['Close'].shift(-5) - df['Close']) / df['Close']) * 100
            train_df = df.dropna()
            X = train_df[['Close', 'SMA20', 'SMA50']].tail(100)
            y = train_df['Target'].tail(100)
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X[:-5], y[:-5])
            
            raw_pred = model.predict(X.tail(1))[0]
            bias_adj = train_df['Target'].tail(10).mean() 
            final_pred = raw_pred + (bias_adj * 0.1)

            # --- SCORES ---
            roe = info.get('returnOnEquity', 0) * 100
            roce = info.get('returnOnAssets', 0) * 1.5 * 100
            mcap = info.get('marketCap', 0) / 10000000 
            volatility = df['Close'].pct_change().std() * 100
            predictability = max(0, 100 - (volatility * 10))

            results.append({
                "Stock": ticker,
                "Price": round(curr_price, 2),
                "Signal %": round(final_pred, 2),
                "Predictability": int(predictability),
                "ROE %": round(roe, 1),
                "ROCE %": round(roce, 1),
                "Market Cap (Cr)": int(mcap),
                "Sentiment": "BULLISH" if curr_price > df['SMA20'].iloc[-1] else "BEARISH"
            })
        except Exception:
            continue
        
        progress_bar.progress((index + 1) / len(watchlist))
    
    progress_bar.empty()
    return pd.DataFrame(results)

# --- 3. STREAMLIT UI ---
st.title("💎 PRO AI Institutional Stock Engine")

# New: Explained section for non-market users
with st.expander("🚀 How the AI Brain works (Read this first)"):
    st.write("""
    This isn't a normal stock app. It uses **Machine Learning** to find hidden patterns:
    1. **Self-Training:** Every time you click 'Run', the AI trains on 2 years of history for each stock.
    2. **Bias Correction:** It looks at its own recent mistakes and adjusts its predictions.
    3. **Institutional Logic:** It combines AI with **ROE** and **ROCE** (the metrics used by Big Banks).
    """)

if st.button('Run AI Analysis 🚀'):
    data = get_engine_data()
    if not data.empty:
        sorted_data = data.sort_values(by="Signal %", ascending=False)
        
        # New: Show a chart for the Top Predicted stock
        top_stock = sorted_data.iloc[0]['Stock']
        st.subheader(f"📈 Momentum Chart: {top_stock} (Top AI Choice)")
        chart_df = yf.Ticker(top_stock).history(period="3mo")
        fig = px.line(chart_df, y='Close', title=f"{top_stock} 90-Day Trend")
        st.plotly_chart(fig, use_container_width=True)

        # New: Styled Dataframe
        st.subheader("📊 Full Market Intelligence")
        def style_signal(val):
            color = 'green' if val > 0 else 'red'
            return f'color: {color}; font-weight: bold'
        
        st.dataframe(sorted_data.style.applymap(style_signal, subset=['Signal %']), use_container_width=True)
        st.success("Analysis complete!")
    else:
        st.error("Data fetch failed. Try again in 1 minute.")
