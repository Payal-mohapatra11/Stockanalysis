import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(page_title="TCS Predictor", layout="wide")
st.title("📈 TCS Next-Day Stock Price Prediction (Daily ML Training)")

# ------------------------------------------------
# 1️⃣ Fetch Latest Stock Data
# ------------------------------------------------
st.info("⏳ Fetching Live Market Data...")

try:
    # Main data fetch
    tcs = yf.Ticker("TCS.NS").history(period="1y", interval="1d")

    # Backup
    if tcs.empty:
        tcs = yf.download("TCS.NS", period="1y", interval="1d", threads=False)

    # Last fallback (direct CSV download)
    if tcs.empty:
        url = "https://query1.finance.yahoo.com/v7/finance/download/TCS.NS?period1=1672531200&period2=9999999999&interval=1d&events=history"
        tcs = pd.read_csv(url)
        tcs.set_index("Date", inplace=True)
        tcs.index = pd.to_datetime(tcs.index)

except Exception as e:
    st.error(f"❌ Could not fetch data: {e}")
    st.stop()

# Check data format
if "Close" not in tcs.columns:
    st.error("❌ 'Close' column missing in fetched data!")
    st.stop()

# Clean data
tcs = tcs[['Close']]
tcs.index = pd.to_datetime(tcs.index)

st.success("✅ Market Data Loaded Successfully!")
st.dataframe(tcs.tail())

# ------------------------------------------------
# 2️⃣ Train Model Daily with Latest Data
# ------------------------------------------------
st.info("🔄 Training model with latest price behavior...")

tcs['Target'] = tcs['Close'].shift(-1)
tcs = tcs.dropna()

X = tcs[['Close']]
y = tcs['Target']

model = LinearRegression()
model.fit(X, y)

st.success("🎯 Model trained successfully with current market data!")

# ------------------------------------------------
# 3️⃣ Predict Tomorrow's Price
# ------------------------------------------------
current_price = float(tcs['Close'].iloc[-1])
predicted_price = float(model.predict([[current_price]])[0])

# Display Results
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
# 4️⃣ Price Chart
# ------------------------------------------------
fig = px.line(tcs['Close'], title="📊 TCS Stock Price (Last 1 Year)", markers=True)
fig.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig, use_container_width=True)

# Final Note
st.info("✨ Model retrains daily on latest market data. No TensorFlow needed.")
