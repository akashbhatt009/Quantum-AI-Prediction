# ⚡ Quantum AI Prediction Terminal

An advanced financial forecasting engine built with Streamlit and Scikit-Learn. This terminal uses a **Recursive Random Forest** approach to predict asset trajectories based on price-action and moving average clusters.

### 🧠 Core Features:
* **Neural Forecasting:** Predicts the asset price 5 days into the future.
* **Self-Correction Logic:** The model performs a real-time **Backtest** on the last 15 valid data windows. It compares its past predictions against actual historical outcomes to generate a "Confidence Score."
* **Probability Zones:** Uses the standard deviation of historical residuals to map an "Error Margin" cloud, visualizing market uncertainty.
* **Sentiment Intelligence:** Integrated AlphaVantage News API to gauge market "mood" and align it with technical indicators.

### 🛠️ How it "Learns":
The app uses **Instance-Based Learning**. Every time you run a search:
1. It fetches the most recent 100 days of data.
2. It retrains the Random Forest Regressor on the fly.
3. It "Self-Corrects" by measuring the delta between its past training iterations and current reality.

<img width="3410" height="1760" alt="image" src="https://github.com/user-attachments/assets/c383630e-3753-4770-bbed-bb7a25b27046" />
