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
    # Read stock symbols from a CSV file. It is expected that the CSV contains a 'Symbol' column.
    stocks_df = pd.read_csv('stocks.csv')  # Ensure you have a 'stocks.csv' file
    stock_symbols = stocks_df['Symbol'].tolist()  # Convert the 'Symbol' column to a list
except Exception as e:
    # Print error message and exit if there is an issue reading the file
    print(f"Error reading stocks.csv: {e}")
    exit()

# --- Load already processed stocks ---
completed_stocks_file = 'completed_stocks.json'  # File to track completed stocks
if os.path.exists(completed_stocks_file):
    # If the file exists, load the list of completed stocks from the JSON file
    with open(completed_stocks_file, 'r') as f:
        completed_stocks = json.load(f)
else:
    # If the file does not exist, initialize an empty list for completed stocks
    completed_stocks = []

# --- Iterate through stocks ---
for stock in stock_symbols:
    # Check if the stock has already been processed
    if stock in completed_stocks:
        print(f"Model for {stock} already trained. Skipping...")
        continue  # Skip to the next stock if already completed
    
    try:
        # Load historical stock price data from Yahoo Finance
        start = '2012-01-01'  # Define the start date for data retrieval
        end = '2024-10-21'    # Define the end date for data retrieval
        data = yf.download(stock, start, end)  # Download the stock data
        data.reset_index(inplace=True)  # Reset the index to make the date a column

        # Prepare training data
        data.dropna(inplace=True)  # Remove any rows with missing values
        data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])  # Use 80% of the data for training

        # Scale the training data to the range [0, 1] using Min-Max scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_train_scale = scaler.fit_transform(data_train)

        # Prepare the input (x) and output (y) datasets for LSTM training
        x = []  # Initialize the list for input features
        y = []  # Initialize the list for target values

        # Create sequences of 100 time steps for input and corresponding targets
        for i in range(100, data_train_scale.shape[0]):
            x.append(data_train_scale[i - 100:i])  # Append the last 100 scaled values as input
            y.append(data_train_scale[i, 0])  # Append the corresponding output value (scaled)

        # Convert the lists to numpy arrays for model training
        x, y = np.array(x), np.array(y)

        # Build the LSTM model
        model = Sequential()  # Initialize a Sequential model
        # Add LSTM layers with dropout for regularization to prevent overfitting
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x.shape[1], 1)))
        model.add(Dropout(0.2))  # Dropout layer with 20% dropout rate

        model.add(LSTM(units=60, activation='relu', return_sequences=True))  # Second LSTM layer
        model.add(Dropout(0.3))  # 30% dropout rate

        model.add(LSTM(units=80, activation='relu', return_sequences=True))  # Third LSTM layer
        model.add(Dropout(0.4))  # 40% dropout rate

        model.add(LSTM(units=120, activation='relu'))  # Fourth LSTM layer
        model.add(Dropout(0.5))  # 50% dropout rate

        model.add(Dense(units=1))  # Output layer for a single prediction

        # Compile the model using Adam optimizer and mean squared error loss function
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model with the prepared data
        model.fit(x, y, epochs=50, batch_size=32, verbose=1)  # Fit the model on the data

        # Save the trained model to a specified directory
        save_dir = f'./data/{stock}'  # Create a directory for the stock
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
        model.save(os.path.join(save_dir, 'Stock_Predictions_Model.keras'))  # Save the model

        print(f"Model for {stock} saved successfully!")  # Confirm successful save

        # Record the stock as completed
        completed_stocks.append(stock)  # Add the stock to the completed list
        with open(completed_stocks_file, 'w') as f:
            json.dump(completed_stocks, f)  # Save the updated completed stocks list to the JSON file

    except Exception as e:
        # Print error message if there is an issue processing the stock
        print(f"Error processing {stock}: {e}")
