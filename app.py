import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import streamlit as st
import plotly.graph_objects as go
import datetime

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

# --- Data Fetching ---
today = datetime.date.today()
try:
    stock = yf.Ticker(stock_symbol) 
    data = stock.history(period="max")
except Exception as e:
    st.error(f"Error fetching data for {stock_symbol}: {e}")
    st.stop()

# --- Stock Information ---
with col2:
    st.subheader('Stock Information')
    info_dict = {
        "Sector": stock.info.get('sector'),
        "Industry": stock.info.get('industry'),
        "Previous Close": stock.info.get('previousClose'),
        "Market Cap": stock.info.get('marketCap'),
        "Trailing P/E": stock.info.get('trailingPE'),
        "Forward P/E": stock.info.get('forwardPE'),
        "Volume": stock.info.get('volume'),
        "Average Volume": stock.info.get('averageVolume'),
        "Dividend Yield": stock.info.get('dividendYield'),
    }
    info_df = pd.DataFrame(list(info_dict.items()), columns=['Attribute', 'Value'])
    st.table(info_df)

# --- Model Training ---
save_dir = f'./data/{stock_symbol}'
model_path = os.path.join(save_dir, 'Stock_Predictions_Model.keras')

if not os.path.exists(model_path):
    st.warning(f"No trained model found for {stock_symbol}. Train a new model.")
    if st.button('Train Model'):
        # Prepare training data
        data.reset_index(inplace=True)
        data.dropna(inplace=True)
        data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])

        scaler = MinMaxScaler(feature_range=(0,1))
        data_train_scale = scaler.fit_transform(data_train)

        x_train = []
        y_train = []

        for i in range(100, data_train_scale.shape[0]):
            x_train.append(data_train_scale[i-100:i])
            y_train.append(data_train_scale[i,0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build and train the model
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Display training progress
        progress_bar = st.progress(0)
        epochs = 50
        for i in range(epochs):
            model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
            progress_bar.progress((i + 1) / epochs)

        os.makedirs(save_dir, exist_ok=True)
        model.save(model_path)
        st.success(f"Model trained and saved for {stock_symbol}!")

# --- Prediction and Visualization ---
if os.path.exists(model_path):
    from keras.models import load_model
    model = load_model(model_path)

    # Prepare test data
    test_start = '2023-01-01'
    test_data = yf.download(stock_symbol, start=test_start, end=today)
    test_data.reset_index(inplace=True)
    actual_prices = test_data['Close'].values

    dataset_total = pd.concat((data['Close'], test_data['Close']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(test_data) - 100:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    x_test = []
    for i in range(100, inputs.shape[0]):
        x_test.append(inputs[i-100:i])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot actual vs. predicted prices
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=test_data['Date'], y=actual_prices, mode='lines', name='Actual Price'))
    fig_pred.add_trace(go.Scatter(x=test_data['Date'], y=predicted_prices.flatten(), mode='lines', name='Predicted Price'))
    fig_pred.update_layout(title=f'{stock_symbol} Price Prediction', xaxis_title='Time', yaxis_title='Price', height=600)
    st.plotly_chart(fig_pred, use_container_width=True)

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
    fig.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Closing Price'))
if show_ma50:
    fig.add_trace(go.Scatter(x=data.index, y=data.Close.rolling(50).mean(), mode='lines', name='MA50'))
if show_ma100:
    fig.add_trace(go.Scatter(x=data.index, y=data.Close.rolling(100).mean(), mode='lines', name='MA100'))
if show_ma200:
    fig.add_trace(go.Scatter(x=data.index, y=data.Close.rolling(200).mean(), mode='lines', name='MA200'))

fig.update_layout(
    title=f'{stock_symbol} Stock Price', 
    xaxis_title='Time', 
    yaxis_title='Price',
    height=600,
    template="plotly_white",
)
