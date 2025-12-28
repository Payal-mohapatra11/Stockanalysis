import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("📈 TCS Next-Day Stock Price Prediction")

# 🔥 Load model
model = load_model("models/tcs_lstm_model.h5")

# 🔥 Fetch LATEST data (includes 2025)
tcs = yf.download("TCS.NS", start="2024-01-01", end=None)
tcs = tcs[['Close']]

# 🔥 Scaling data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(tcs)

# 🔥 Last 60 days for Prediction
last_60 = scaled_data[-60:].reshape(1,60,1)
pred_scaled = model.predict(last_60)
predicted_price = scaler.inverse_transform(pred_scaled)[0][0]

# 🔥 Current price
current_price = tcs['Close'].iloc[-1]

# 🎯 Display Results
st.subheader("📌 Latest TCS Close Price:")
st.write(round(current_price,2))

st.subheader("🔮 Predicted Next Day Price:")
st.write(round(predicted_price,2))

# 🔁 Trend Direction
if predicted_price > current_price:
    st.success("🚀 **Uptrend Expected — BUY Signal (not financial advice)**")
else:
    st.error("📉 **Downtrend Expected — SELL Signal (not financial advice)**")

# 📊 Chart
st.line_chart(tcs['Close'])
