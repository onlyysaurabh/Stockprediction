import os
import json
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import logging
from pymongo import MongoClient

# Configure logging
logging.basicConfig(filename='stock_prediction.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration (could be loaded from a separate file) ---
START_DATE = '2015-02-06'  # Consider making this configurable
END_DATE = '2025-02-06'    # Consider making this configurable
TRAIN_SPLIT = 0.80
SEQUENCE_LENGTH = 60  # Adjusted based on the code
EPOCHS = 50  # Updated based on the updated response
BATCH_SIZE = 32
STOCKS_FILE = 'stocks.csv'
COMPLETED_STOCKS_FILE = 'completed_stocks.json'
BASE_SAVE_DIR = './data'

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB connection string
DB_NAME = "stock_market_data"  # Replace with your database name
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# --- Load stock symbols ---
try:
    stocks_df = pd.read_csv(STOCKS_FILE)
    stock_symbols = stocks_df['Symbol'].tolist()
    logging.info(f"Loaded {len(stock_symbols)} stock symbols from {STOCKS_FILE}")
except Exception as e:
    logging.error(f"Error reading {STOCKS_FILE}: {e}")
    exit()

# --- Load already processed stocks ---
if os.path.exists(COMPLETED_STOCKS_FILE):
    with open(COMPLETED_STOCKS_FILE, 'r') as f:
        completed_stocks = json.load(f)
else:
    completed_stocks =

# --- Iterate through stocks ---
for stock in stock_symbols:
    if stock in completed_stocks:
        logging.info(f"Models for {stock} already trained. Skipping...")
        continue

    try:
        # Download historical stock data using yfinance
        data = yf.download(stock, start=START_DATE, end=END_DATE)
        if data.empty:
            logging.warning(f"No data found for {stock}, skipping.")
            continue
        data.reset_index(inplace=True)

        # --- Load news sentiment data ---
        news_sentiment_df = pd.DataFrame(list(db['news_data'].find({'symbol': stock})))
        if news_sentiment_df.empty:
            logging.warning(f"No news sentiment data found for {stock}, skipping.")
            continue
        news_sentiment_df['Date'] = pd.to_datetime(news_sentiment_df['date'])
        news_sentiment_df = news_sentiment_df.groupby('Date')['sentiment'].mean().reset_index()

        # --- Load commodity price data (example: Gold) ---
        gold_prices_df = pd.DataFrame(list(db['commodities_data'].find({'Name': 'Gold'})))
        if gold_prices_df.empty:
            logging.warning(f"No gold price data found, skipping.")
            continue
        gold_prices_df['Date'] = pd.to_datetime(gold_prices_df['Date'])

        # --- Combine data and create lagged features ---
        data = data.merge(news_sentiment_df, on='Date', how='left')
        data = data.merge(gold_prices_df[['Date', 'Close']], on='Date', how='left', suffixes=('', '_Gold'))
        data['Close_Lag1'] = data['Close'].shift(1)
        #... add more lagged features or commodity prices as needed...

        # --- Prepare training data ---
        data.dropna(inplace=True)
        features = ['Close', 'sentiment', 'Close_Gold', 'Close_Lag1']  # Include all features
        data_scaled = pd.DataFrame(MinMaxScaler(feature_range=(0, 1)).fit_transform(data[features]), columns=features)

        # --- Create the training data set ---
        training_data_len = int(np.ceil(len(data_scaled) * TRAIN_SPLIT))
        train_data = data_scaled[0:training_data_len]

        # --- Split the data into x_train and y_train data sets ---
        x_train =
        y_train =
        for i in range(SEQUENCE_LENGTH, len(train_data)):
            x_train.append(train_data[features].values[i - SEQUENCE_LENGTH:i])  # Use all features
            y_train.append(train_data['Close'][i])
        x_train, y_train = np.array(x_train), np.array(y_train)

        # --- Reshape for LSTM ---
        x_train_lstm = np.reshape(x_train, (x_train.shape, x_train.shape, len(features)))

        # --- Build the LSTM model ---
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_lstm.shape, len(features))))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # --- Compile the model ---
        model.compile(optimizer='adam', loss='mean_squared_error')

        # --- Add early stopping callback ---
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # --- Create the testing data set ---
        test_data = data_scaled[training_data_len - SEQUENCE_LENGTH:]
        x_test =
        y_test = data_scaled['Close'][training_data_len:]

        for i in range(SEQUENCE_LENGTH, len(test_data)):
            x_test.append(test_data[features].values[i - SEQUENCE_LENGTH:i])  # Use all features

        x_test = np.array(x_test)
        y_test = np.array(y_test)
        # Reshape x_test for LSTM
        x_test_lstm = np.reshape(x_test, (x_test.shape, x_test.shape, len(features)))

        # --- Train the LSTM model with validation data and early stopping ---
        model.fit(x_train_lstm, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                  validation_data=(x_test_lstm, y_test), callbacks=[early_stopping], verbose=1)

        # --- Train and evaluate other models (SVM, Random Forest, XGBoost) ---
        # (Code for training and evaluating SVM, Random Forest, and XGBoost models remains the same)

        # --- Save the trained models ---
        save_dir = os.path.join(BASE_SAVE_DIR, stock)
        os.makedirs(save_dir, exist_ok=True)

        model.save(os.path.join(save_dir, 'Stock_Predictions_Model.keras'))
        with open(os.path.join(save_dir, 'svm_model.pkl'), 'wb') as f:
            pickle.dump(svm_model, f)
        with open(os.path.join(save_dir, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(rf_model, f)
        with open(os.path.join(save_dir, 'xgb_model.pkl'), 'wb') as f:
            pickle.dump(xgb_model, f)

        logging.info(f"Models for {stock} saved successfully!")

        # Record the stock as completed
        completed_stocks.append(stock)
        with open(COMPLETED_STOCKS_FILE, 'w') as f:
            json.dump(completed_stocks, f)

    except Exception as e:
        logging.error(f"Error training model for {stock}: {e}")