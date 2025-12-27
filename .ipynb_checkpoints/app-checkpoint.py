import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("📈 TCS Stock Price Prediction App")

# Load model
model = load_model("models/tcs_lstm_model.h5")

# Load data
df = pd.read_csv("data/NIFTY_50_COMPANIES.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Ticker'] == "TCS.NS"]
df = df[['Date','Close']]
df = df.dropna()
df = df.set_index('Date')

# Scaling
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df[['Close']])

# Last 60 days
last_60 = scaled_data[-60:].reshape(1,60,1)
pred_scaled = model.predict(last_60)
predicted_price = scaler.inverse_transform(pred_scaled)[0][0]

# Current price
current_price = df['Close'].iloc[-1]

# Results
st.write("### 🔮 Predicted Next Day Price:", round(predicted_price,2))
st.write("### 📌 Current Price:", current_price)

if predicted_price > current_price:
    st.write("### 📈 Trend: **UP (Bullish)** 🚀")
else:
    st.write("### 📉 Trend: **DOWN (Bearish)** 🔻")

# Chart
st.line_chart(df['Close'])

