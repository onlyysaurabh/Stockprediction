import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pymongo import MongoClient
from keras.models import load_model

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]
stock_data_collection = db["stock_data"]
stock_info_collection = db["stock_info"]
commodities_collection = db["commodities_data"]  # Collection for commodity prices

# --- App Styling ---
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        padding: 2rem;
    }
    .stTable {
        padding: 0px !important; 
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
model = None
if os.path.exists(model_path):
    model = load_model(model_path)
    st.success("Model loaded successfully!")
else:
    st.warning(f"No saved model found for {stock_symbol}.")

# --- Stock Information ---
with col2:
    st.subheader('Stock Information')
    stock_info_from_db = stock_info_collection.find_one({"symbol": stock_symbol})
    if stock_info_from_db:
        st.success("Stock info loaded from MongoDB!")
        info_df = pd.DataFrame(list(stock_info_from_db.items()), columns=['Attribute', 'Value'])
        info_df = info_df[info_df['Attribute'] != '_id']
        st.table(info_df)
    else:
        st.error(f"No stock information found for {stock_symbol} in MongoDB.")

# --- Historical Data Chart ---
st.subheader('Select Parameters for Historical Data')
col3, col4, col5, col6, col7 = st.columns(5)  # Added columns for each checkbox
with col3:
    show_close = st.checkbox('Closing Price', value=True)
with col4:
    show_ma50 = st.checkbox('MA50', value=True)
with col5:
    show_commodity_oil = st.checkbox('Oil (WTI Crude)', value=False)
with col6:
    show_commodity_gold = st.checkbox('Gold', value=False)
with col7:
    show_commodity_bitcoin = st.checkbox('Bitcoin', value=False)

# Plotting the stock price and moving averages
fig = go.Figure()
if show_close:
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price'))
if show_ma50:
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(50).mean(), mode='lines', name='MA50'))

# --- Add Commodity Price Data ---
commodities_to_check = []
if show_commodity_oil:
    commodities_to_check.append('Oil (WTI Crude)')
if show_commodity_gold:
    commodities_to_check.append('Gold')
if show_commodity_bitcoin:
    commodities_to_check.append('Bitcoin')

for commodity in commodities_to_check:
    commodity_data = commodities_collection.find_one({"Commodity": commodity})
    if commodity_data:
        commodity_df = pd.DataFrame(commodity_data["Data"])
        commodity_df['Date'] = pd.to_datetime(commodity_df['Date'])
        commodity_df.set_index('Date', inplace=True)
        fig.add_trace(go.Scatter(x=commodity_df.index, y=commodity_df['Close'], mode='lines', name=commodity))
    else:
        st.warning(f"No data found for {commodity} in MongoDB.")

# Finalize the layout for historical data chart
fig.update_layout(
    title=f'{stock_symbol} Stock Price',
    xaxis_title='Time',
    yaxis_title='Price',
    height=600,
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)

# --- Sentiment and Stock Price Graphs ---
st.subheader('Sentiment and Stock Price Correlation')

try:
    news_from_db = db["news_data"].find({"symbol": stock_symbol})
    news_df = pd.DataFrame(list(news_from_db))
    if not news_df.empty:
        news_df = news_df.drop(columns=['_id', 'symbol'])
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_df = news_df.sort_values(by='date')

        # Use only the dates from the sentiment data
        sentiment_dates = news_df['date']
        filtered_data = data[data.index.isin(sentiment_dates)]

        # Create subplots
        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(go.Scatter(x=news_df['date'], y=news_df['sentiment'], mode='lines', name='Sentiment Score'))
        fig_sentiment.update_layout(
            title='Sentiment Score Over Time',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            template="plotly_white",
        )

        fig_stock = go.Figure()
        fig_stock.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Close'], mode='lines', name='Closing Price'))
        fig_stock.update_layout(
            title=f'{stock_symbol} Stock Price Over Time',
            xaxis_title='Date',
            yaxis_title='Price',
            template="plotly_white",
        )

        # Display plots side by side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_sentiment, use_container_width=True)
        with col2:
            st.plotly_chart(fig_stock, use_container_width=True)

except Exception as e:
    st.error(f"Error fetching sentiment data: {e}")

# --- News and Sentiment Display ---
st.subheader('Recent News and Sentiment')
try:
    news_from_db = db["news_data"].find({"symbol": stock_symbol})
    news_df = pd.DataFrame(list(news_from_db))
    if not news_df.empty:
        news_df = news_df.drop(columns=['_id', 'symbol'])
        news_df['date'] = pd.to_datetime(news_df['date']).dt.strftime("%B %d, %Y")

        # Create an "Open" hyperlink column
        news_df['Open'] = news_df['link'].apply(lambda link: f"<a href='{link}' target='_blank'>Open</a>")
        news_df = news_df.drop(columns=['link'])  # Remove the original link column

        # Display news in a regular Streamlit table with "Open" hyperlinks
        st.write(news_df.style.format({"sentiment": "{:.2f}"})
                 .applymap(lambda x: "color: green" if (isinstance(x, float) and x > 0) or (isinstance(x, str) and x.startswith('green')) 
                           else "color: red" if (isinstance(x, float) and x < 0) or (isinstance(x, str) and x.startswith('red'))
                           else "", subset=['sentiment'])
                 .to_html(escape=False, index=False),
                 unsafe_allow_html=True)

    else:
        st.write("No recent news found for this stock.")
except Exception as e:
    st.error(f"Error fetching/displaying news: {e}")
