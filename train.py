import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

# --- Load stock symbols ---
try:
    stocks_df = pd.read_csv('stocks.csv')  # Ensure you have a 'stocks.csv' file
    stock_symbols = stocks_df['Symbol'].tolist()
except Exception as e:
    print(f"Error reading stocks.csv: {e}")
    exit()

# --- Load already processed stocks ---
completed_stocks_file = 'completed_stocks.json'
if os.path.exists(completed_stocks_file):
    with open(completed_stocks_file, 'r') as f:
        completed_stocks = json.load(f)
else:
    completed_stocks = []

# --- Iterate through stocks ---
for stock in stock_symbols:
    if stock in completed_stocks:
        print(f"Model for {stock} already trained. Skipping...")
        continue
    
    try:
        # Load data
        start = '2012-01-01'
        end = '2024-10-21'
        data = yf.download(stock, start, end)
        data.reset_index(inplace=True)

        # Prepare training data
        data.dropna(inplace=True)
        data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_train_scale = scaler.fit_transform(data_train)

        x = []
        y = []

        for i in range(100, data_train_scale.shape[0]):
            x.append(data_train_scale[i - 100:i])
            y.append(data_train_scale[i, 0])

        x, y = np.array(x), np.array(y)

        # Build and train the model
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x, y, epochs=50, batch_size=32, verbose=1)

        # Save the model
        save_dir = f'./data/{stock}'
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, 'Stock_Predictions_Model.keras'))

        print(f"Model for {stock} saved successfully!")

        # Record the completed stock
        completed_stocks.append(stock)
        with open(completed_stocks_file, 'w') as f:
            json.dump(completed_stocks, f)

    except Exception as e:
        print(f"Error processing {stock}: {e}")