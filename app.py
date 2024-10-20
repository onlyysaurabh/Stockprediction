import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import datetime
from pymongo import MongoClient
from keras.models import load_model

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]
stock_data_collection = db["stock_data"]

# --- App Styling ---
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title('Stock Market Analyzer')

# --- Stock Selection ---
try:
    stocks_df = pd.read_csv('stocks.csv')
    stock_symbols = stocks_df['Symbol'].tolist()
except Exception as e:
    st.error(f"Error reading stocks.csv: {e}")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    stock_symbol = st.selectbox('Select Stock', stock_symbols)

# --- Data Fetching (from MongoDB) ---
data = None
try:
    data_from_db = stock_data_collection.find_one({"Symbol": stock_symbol})
    if data_from_db:
        data = pd.DataFrame(data_from_db["Data"])
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        st.success("Data loaded from MongoDB!")
    else:
        st.error(f"No data found for {stock_symbol} in MongoDB.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching/loading data for {stock_symbol}: {e}")
    st.stop()

# --- Load Model ---
model_path = f'./data/{stock_symbol}/Stock_Predictions_Model.keras'
if os.path.exists(model_path):
    model = load_model(model_path)
    st.success("Model loaded successfully!")
else:
    st.error(f"No saved model found for {stock_symbol}.")

# --- Stock Information ---
with col2:
    st.subheader('Stock Information')
    try:
        stock_info = {
            "Sector": "N/A",  # Placeholder if not fetched
            "Industry": "N/A",
            "Previous Close": "N/A",
            "Market Cap": "N/A",
            "Trailing P/E": "N/A",
            "Forward P/E": "N/A",
            "Volume": "N/A",
            "Average Volume": "N/A",
            "Dividend Yield": "N/A",
        }
        info_df = pd.DataFrame(list(stock_info.items()), columns=['Attribute', 'Value'])
        st.table(info_df)
    except Exception as e:
        st.error(f"Error retrieving stock information: {e}")

# --- Historical Data Chart ---
st.subheader('Select Parameters for Historical Data')
col3, col4 = st.columns(2)
with col3:
    show_close = st.checkbox('Closing Price', value=True)
    show_ma50 = st.checkbox('MA50')
with col4:
    show_ma100 = st.checkbox('MA100')
    show_ma200 = st.checkbox('MA200')

# --- Chart for Historical Data ---
fig = go.Figure()
if show_close:
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price'))
if show_ma50:
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(50).mean(), mode='lines', name='MA50'))
if show_ma100:
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(100).mean(), mode='lines', name='MA100'))
if show_ma200:
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(200).mean(), mode='lines', name='MA200'))

fig.update_layout(
    title=f'{stock_symbol} Stock Price', 
    xaxis_title='Time', 
    yaxis_title='Price',
    height=600,
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)

# --- News Display ---
st.subheader('Recent News')
try:
    news_from_db = db["news_data"].find({"symbol": stock_symbol})
    if news_from_db:
        news_df = pd.DataFrame(list(news_from_db))
        st.write(news_df)
    else:
        st.write("No recent news found for this stock.")
except Exception as e:
    st.error(f"Error fetching/displaying news: {e}")
