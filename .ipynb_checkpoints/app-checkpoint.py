import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

st.set_page_config(page_title="TCS Predictor", layout="wide")
st.title("📈 TCS Next-Day Stock Price Prediction (Sklearn Model)")

# ------------------------------------------------
# 1️⃣ Load Model
# ------------------------------------------------
try:
    model = joblib.load("models/tcs_sklearn_model.pkl")
    st.success("📌 Model Loaded Successfully!")
except:
    st.error("❌ Model file missing! Put `tcs_sklearn_model.pkl` inside `/models/` folder.")
    st.stop()

# ------------------------------------------------
# 2️⃣ Fetch Latest Stock Data
# ------------------------------------------------
st.info("⏳ Fetching Live Market Data...")

try:
    # Main fetch
    tcs = yf.Ticker("TCS.NS").history(period="1y", interval="1d")

    # Backup method
    if tcs.empty:
        tcs = yf.download("TCS.NS", period="1y", interval="1d", threads=False)

    # Final fallback
    if tcs.empty:
        url = "https://query1.finance.yahoo.com/v7/finance/download/TCS.NS?period1=1672531200&period2=9999999999&interval=1d&events=history"
        tcs = pd.read_csv(url)
        tcs.set_index("Date", inplace=True)
        tcs.index = pd.to_datetime(tcs.index)

except Exception as e:
    st.error(f"❌ Could not fetch data: {e}")
    st.stop()

# Check "Close" column
if "Close" not in tcs.columns:
    st.error("❌ 'Close' column missing! Data format error.")
    st.stop()

tcs = tcs[['Close']]
tcs.index = pd.to_datetime(tcs.index)

st.success("✅ Data Loaded Successfully!")
st.dataframe(tcs.tail())

# ------------------------------------------------
# 3️⃣ Scale + Predict Next Day
# ------------------------------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(tcs)

# Prepare last value for prediction (Sklearn: 2D input)
latest_value = scaled[-1].reshape(1, -1)

# Predict
pred_scaled = model.predict(latest_value)
predicted_price = float(scaler.inverse_transform([[pred_scaled[0]]])[0][0])

current_price = float(tcs['Close'].iloc[-1])

# ------------------------------------------------
# 4️⃣ Display Results
# ------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Today's Close Price")
    st.write(f"### ₹ {round(current_price,2)}")

with col2:
    st.subheader("🔮 Predicted Tomorrow Price")
    st.write(f"### ₹ {round(predicted_price,2)}")

st.write("---")

# Trend Signal
if predicted_price > current_price:
    st.success("🚀 **Uptrend Expected — BUY Signal** *(Not financial advice)*")
else:
    st.error("📉 **Downtrend Expected — SELL Signal** *(Not financial advice)*")

# ------------------------------------------------
# 5️⃣ Line Chart
# ------------------------------------------------
fig = px.line(tcs, y='Close', title="📊 TCS Stock Price (Last 1 Year)", markers=True)
fig.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig, use_container_width=True)

st.info("✨ This ML model predicts next trading day's price using latest market data.")
