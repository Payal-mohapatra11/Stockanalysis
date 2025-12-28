import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
#from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

st.set_page_config(page_title="TCS Predictor", layout="wide")
st.title("📈 TCS Next-Day Stock Price Prediction (LSTM Model)")

# ------------------------------------------------
# 1️⃣ Load Model
# ------------------------------------------------
try:
   # model = load_model("models/tcs_lstm_model.h5")
    # from tensorflow.keras.models import load_model
st.warning("⚠️ Prediction temporarily disabled. TensorFlow setup in progress.")
pred = None
except:
    st.error("❌ Model file missing! Put `tcs_lstm_model.h5` inside `/models/` folder.")
    st.stop()

# ------------------------------------------------
# 2️⃣ Fetch Latest Stock Data
# ------------------------------------------------
st.info("⏳ Fetching Live Market Data...")

try:
    # Try main method
    tcs = yf.Ticker("TCS.NS").history(period="1y", interval="1d")

    # Fallback
    if tcs.empty:
        tcs = yf.download("TCS.NS", period="1y", interval="1d", threads=False)

    # Last fallback using direct CSV link
    if tcs.empty:
        url = "https://query1.finance.yahoo.com/v7/finance/download/TCS.NS?period1=1704067200&period2=9999999999&interval=1d&events=history"
        tcs = pd.read_csv(url)
        tcs.set_index("Date", inplace=True)
        tcs.index = pd.to_datetime(tcs.index)
    
except Exception as e:
    st.error(f"❌ Could not fetch data: {e}")
    st.stop()

# Keep only closing prices
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

if len(scaled) < 60:
    st.error("❌ Not enough data to predict (need minimum 60 days).")
    st.stop()

last_60 = scaled[-60:].reshape(1,60,1)
pred_scaled = model.predict(last_60)
predicted_price = float(scaler.inverse_transform(pred_scaled)[0][0])
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
# 5️⃣ Chart
# ------------------------------------------------
fig = px.line(tcs, y='Close', title="📊 TCS Stock Price (Last 1 Year)", markers=True)
fig.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig, use_container_width=True)

st.info("✨ Model uses last 60 days to predict the next trading day's price.")
