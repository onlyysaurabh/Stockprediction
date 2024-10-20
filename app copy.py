import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
import streamlit as st
import plotly.graph_objects as go
import datetime
from pymongo import MongoClient
from GoogleNews import GoogleNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dateutil.relativedelta import relativedelta
from transformers import pipeline

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]
stock_data_collection = db["stock_data"]
news_data_collection = db["news_data"]

# --- FinBERT Pipeline ---
classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')

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
st.title('Stock Market Analyzer & Predictor')

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

# --- Data Fetching (from MongoDB or yfinance) ---
today = datetime.date.today()
data = None
stock = yf.Ticker(stock_symbol)  # Ensure this is initialized

try:
    data_from_db = stock_data_collection.find_one({"Symbol": stock_symbol})
    if data_from_db:
        data = pd.DataFrame(data_from_db["Data"])
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        st.success("Data loaded from MongoDB!")
    else:
        data = stock.history(period="max")
        stock_data_collection.insert_one({"Symbol": stock_symbol, "Data": data.reset_index().to_dict("records")})
        st.success("Data fetched from yfinance and stored in MongoDB!")

except Exception as e:
    st.error(f"Error fetching/loading data for {stock_symbol}: {e}")
    st.stop()

# --- Stock Information ---
with col2:
    st.subheader('Stock Information')
    try:
        info_dict = {
            "Sector": stock.info.get('sector', 'N/A'),
            "Industry": stock.info.get('industry', 'N/A'),
            "Previous Close": stock.info.get('previousClose', 'N/A'),
            "Market Cap": stock.info.get('marketCap', 'N/A'),
            "Trailing P/E": stock.info.get('trailingPE', 'N/A'),
            "Forward P/E": stock.info.get('forwardPE', 'N/A'),
            "Volume": stock.info.get('volume', 'N/A'),
            "Average Volume": stock.info.get('averageVolume', 'N/A'),
            "Dividend Yield": stock.info.get('dividendYield', 'N/A'),
        }
        info_df = pd.DataFrame(list(info_dict.items()), columns=['Attribute', 'Value'])
        st.table(info_df)
    except Exception as e:
        st.error(f"Error retrieving stock information: {e}")

# --- Model Training ---
save_dir = f'./data/{stock_symbol}'
model_path = os.path.join(save_dir, 'Stock_Predictions_Model.keras')

if not os.path.exists(model_path):
    st.warning(f"No trained model found for {stock_symbol}. Train a new model.")
    if st.button('Train Model'):
        # Prepare training data
        data.reset_index(inplace=True)
        data.dropna(inplace=True)
        data_train = data[['Date', 'Close']][0: int(len(data)*0.80)]

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_train['Scaled_Close'] = scaler.fit_transform(data_train['Close'].values.reshape(-1, 1))

        # Fetch and prepare news sentiment data
        news_data = fetch_and_preprocess_news(stock_symbol, stock.info.get('longName'), data_train['Date'].min(), data_train['Date'].max())
        
        if news_data is not None:
            data_train = pd.merge(data_train, news_data, on='Date', how='left')
            data_train['Sentiment'].fillna(0, inplace=True)

            # Prepare training data with sentiment
            x_train, y_train = [], []
            for i in range(100, len(data_train)):
                x_train.append(data_train[['Scaled_Close', 'Sentiment']].iloc[i-100:i].values)
                y_train.append(data_train['Scaled_Close'].iloc[i])

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 2)

            # Build and train the model
            model = Sequential()
            model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 2)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=60, activation='relu', return_sequences=True))
            model.add(Dropout(0.3))
            model.add(LSTM(units=80, activation='relu', return_sequences=True))
            model.add(Dropout(0.4))
            model.add(LSTM(units=120, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(units=1))

            model.compile(optimizer='adam', loss='mean_squared_error')

            progress_bar = st.progress(0)
            epochs = 50
            for i in range(epochs):
                model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
                progress_bar.progress((i + 1) / epochs)

            os.makedirs(save_dir, exist_ok=True)
            model.save(model_path)
            st.success(f"Model trained and saved for {stock_symbol}!")
        else:
            st.error("Error fetching news data for training.")

# --- Prediction and Visualization ---
if os.path.exists(model_path):
    model = load_model(model_path)

    # Prepare test data
    test_start = '2023-01-01'
    test_data = yf.download(stock_symbol, start=test_start, end=today)
    test_data.reset_index(inplace=True)
    actual_prices = test_data['Close'].values

    dataset_total = pd.concat((data['Close'], test_data['Close']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(test_data) - 100:].values.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    # Fetch and prepare news sentiment data for the test period
    news_data_test = fetch_and_preprocess_news(stock_symbol, stock.info.get('longName'), test_data['Date'].min(), test_data['Date'].max())
    if news_data_test is not None:
        test_data = pd.merge(test_data, news_data_test, on='Date', how='left')
        test_data['Sentiment'].fillna(0, inplace=True)

        x_test = []
        for i in range(100, len(test_data)):
            x_test.append(np.column_stack((inputs[i-100:i], np.array(test_data['Sentiment'].iloc[i-100:i]).reshape(-1, 1))))
        x_test = np.array(x_test)

        # Make predictions
        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        # Plot actual vs. predicted prices
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=test_data['Date'].iloc[100:], y=actual_prices[100:], mode='lines', name='Actual Price'))
        fig_pred.add_trace(go.Scatter(x=test_data['Date'].iloc[100:], y=predicted_prices.flatten(), mode='lines', name='Predicted Price'))
        fig_pred.update_layout(title=f'{stock_symbol} Price Prediction', xaxis_title='Time', yaxis_title='Price', height=600)
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.error("Error fetching news data for prediction.")

# --- Parameter Selection for Historical Data Chart ---
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

# --- News and Sentiment Display ---
st.subheader('Recent News and Sentiment')
try:
    news_from_db = news_data_collection.find({"symbol": stock_symbol})
    if news_from_db:
        news_df = pd.DataFrame(list(news_from_db))
        st.write(news_df)
except Exception as e:
    st.error(f"Error fetching/displaying news sentiment: {e}")

# --- Helper function to fetch and preprocess news data ---
def fetch_and_preprocess_news(stock_symbol, stock_name, start_date, end_date):
    try:
        googlenews = GoogleNews(lang='en')
        googlenews.set_time_range(start_date.strftime("%m/%d/%Y"), end_date.strftime("%m/%d/%Y"))

        search_query = f"{stock_symbol} {stock_name}"
        googlenews.search(search_query)
        news_results = googlenews.results()

        news_data = []
        for news in news_results:
            title = news['title']
            date_str = news['date']
            link = news['link']

            # Parse the date string and convert to datetime object
            try:
                parts = date_str.split()
                if len(parts) == 3 and parts[1] in ('hour', 'hours', 'minute', 'minutes', 'day', 'days'):
                    num_units = int(parts[0])
                    unit = parts[1].rstrip('s') 

                    if unit in ('hour', 'hours'):
                        date = datetime.datetime.now() - relativedelta(hours=num_units)
                    elif unit in ('minute', 'minutes'):
                        date = datetime.datetime.now() - relativedelta(minutes=num_units)
                    elif unit in ('day', 'days'):
                        date = datetime.datetime.now() - relativedelta(days=num_units)
                    else:
                        date = None
                else:
                    date = datetime.datetime.strptime(date_str, "%b %d, %Y")

                formatted_date = date.strftime("%Y-%m-%dT%H:%M:%S.000+00:00") if date else None

            except ValueError:
                print(f"Error parsing date: {date_str}")
                formatted_date = None

            # Get sentiment score using FinBERT
            try:
                result = classifier(title)[0]
                sentiment = result['score']
            except Exception as e:
                print(f"Error getting sentiment for '{title}': {e}")
                sentiment = 0  # Assign a neutral sentiment if there's an error

            news_data.append({
                "symbol": stock_symbol,
                "date": formatted_date,
                "title": title,
                "link": link,
                "sentiment": sentiment
            })

        # Preprocess news data
        news_df = pd.DataFrame(news_data)
        news_df['Date'] = pd.to_datetime(news_df['date'])
        news_df = news_df.groupby('Date')['sentiment'].mean().reset_index()  # Calculate average daily sentiment
        return news_df

    except Exception as e:
        print(f"Error fetching/analyzing news for {stock_symbol}: {e}")
        return None
