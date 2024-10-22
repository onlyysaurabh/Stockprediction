# Import necessary libraries
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pymongo import MongoClient
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- MongoDB Connection ---
# Define the MongoDB connection URI
MONGO_URI = "mongodb://localhost:27017/"
# Create a MongoDB client
client = MongoClient(MONGO_URI)
# Access the stock market database
db = client["stock_market_db"]
# Define collections for stock data, stock info, and commodities data
stock_data_collection = db["stock_data"]
stock_info_collection = db["stock_info"]
commodities_collection = db["commodities_data"]

# --- App Styling ---
# Set the layout for the Streamlit app
st.set_page_config(layout="wide")
# Add custom CSS for styling the app
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;  /* Maximum width of the app */
        padding: 2rem;      /* Padding around the app */
    }
    .stTable {
        padding: 0px !important;  /* No padding for tables */
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
# Display the title of the app
st.title('Stock Market Analyzer')

# --- Stock Selection ---
# Load stock symbols from a CSV file
try:
    stocks_df = pd.read_csv('stocks.csv')
    stock_symbols = stocks_df['Symbol'].tolist()  # Convert symbols to a list
except Exception as e:
    # Display an error message if loading fails
    st.error(f"Error reading stocks.csv: {e}")
    st.stop()  # Stop execution

# Create columns for layout
col1, col2 = st.columns([1, 2])
with col1:
    # Create a dropdown for selecting a stock
    stock_symbol = st.selectbox('Select Stock', stock_symbols)

# --- Data Fetching (from MongoDB) ---
try:
    # Fetch data for the selected stock from MongoDB
    data_from_db = stock_data_collection.find_one({"Symbol": stock_symbol})
    if data_from_db:
        # Convert the retrieved data into a DataFrame
        data = pd.DataFrame(data_from_db["Data"])
        # Convert the 'Date' column to datetime format
        data['Date'] = pd.to_datetime(data['Date'])
        # Set 'Date' as the index of the DataFrame
        data.set_index('Date', inplace=True)
        st.success("Data loaded from MongoDB!")  # Success message
    else:
        # Display an error if no data is found
        st.error(f"No data found for {stock_symbol} in MongoDB.")
        st.stop()  # Stop execution
except Exception as e:
    # Display an error message if fetching/loading fails
    st.error(f"Error fetching/loading data for {stock_symbol}: {e}")
    st.stop()  # Stop execution

# --- Load Model ---
# Define the path to the saved model
model_path = f'./data/{stock_symbol}/Stock_Predictions_Model.keras'
model = None  # Initialize the model variable
if os.path.exists(model_path):
    # Load the model if it exists
    model = load_model(model_path)
    st.success("Model loaded successfully!")  # Success message
else:
    st.warning(f"No saved model found for {stock_symbol}.")  # Warning if model not found

# --- Stock Information ---
with col2:
    # Display stock information
    st.subheader('Stock Information')
    stock_info_from_db = stock_info_collection.find_one({"symbol": stock_symbol})
    if stock_info_from_db:
        st.success("Stock info loaded from MongoDB!")  # Success message
        # Convert stock info to a DataFrame
        info_df = pd.DataFrame(list(stock_info_from_db.items()), columns=['Attribute', 'Value'])
        info_df = info_df[info_df['Attribute'] != '_id']  # Exclude MongoDB's internal ID field
        st.table(info_df)  # Display the stock information table
    else:
        st.error(f"No stock information found for {stock_symbol} in MongoDB.")

# --- Historical Data Chart ---
st.subheader('Select Parameters for Historical Data')
# Create columns for checkboxes to select data to display
col3, col4, col5, col6, col7 = st.columns(5)
with col3:
    show_close = st.checkbox('Closing Price', value=True)  # Checkbox for closing price
with col4:
    show_ma50 = st.checkbox('MA50', value=True)  # Checkbox for 50-day moving average
with col5:
    show_commodity_oil = st.checkbox('Oil (WTI Crude)', value=False)  # Checkbox for oil prices
with col6:
    show_commodity_gold = st.checkbox('Gold', value=False)  # Checkbox for gold prices
with col7:
    show_commodity_bitcoin = st.checkbox('Bitcoin', value=False)  # Checkbox for bitcoin prices

# Plotting the stock price and moving averages
fig = go.Figure()
if show_close:
    # Add closing price to the plot if selected
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price'))
if show_ma50:
    # Add the 50-day moving average to the plot if selected
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(50).mean(), mode='lines', name='MA50'))

# --- Add Commodity Price Data ---
# Initialize a list to hold commodities to check
commodities_to_check = []
if show_commodity_oil:
    commodities_to_check.append('Oil (WTI Crude)')  # Add oil to the list if selected
if show_commodity_gold:
    commodities_to_check.append('Gold')  # Add gold to the list if selected
if show_commodity_bitcoin:
    commodities_to_check.append('Bitcoin')  # Add bitcoin to the list if selected

# Fetch and plot commodity prices
for commodity in commodities_to_check:
    commodity_data = commodities_collection.find_one({"Commodity": commodity})
    if commodity_data:
        # Convert the commodity data into a DataFrame
        commodity_df = pd.DataFrame(commodity_data["Data"])
        # Convert the 'Date' column to datetime format
        commodity_df['Date'] = pd.to_datetime(commodity_df['Date'])
        commodity_df.set_index('Date', inplace=True)  # Set 'Date' as the index
        # Add the commodity closing price to the plot
        fig.add_trace(go.Scatter(x=commodity_df.index, y=commodity_df['Close'], mode='lines', name=commodity))
    else:
        st.warning(f"No data found for {commodity} in MongoDB.")  # Warning if no data found

# Finalize the layout for historical data chart
fig.update_layout(
    title=f'{stock_symbol} Stock Price',
    xaxis_title='Time',  # X-axis label
    yaxis_title='Price',  # Y-axis label
    height=600,  # Height of the plot
    template="plotly_white",  # Template for the plot's appearance
)

# Display the plot in the app
st.plotly_chart(fig, use_container_width=True)

# --- Sentiment Graph ---
st.subheader('Sentiment Over Time')
try:
    # Fetch news data from MongoDB related to the selected stock
    news_from_db = db["news_data"].find({"symbol": stock_symbol})
    news_df = pd.DataFrame(list(news_from_db))  # Convert news data to a DataFrame
    if not news_df.empty:
        # Drop unnecessary columns and convert 'date' to datetime
        news_df = news_df.drop(columns=['_id', 'symbol'])
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_df = news_df.sort_values(by='date')  # Sort news by date

        # Create sentiment plot
        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(go.Scatter(x=news_df['date'], y=news_df['sentiment'], mode='lines', name='Sentiment Score'))
        fig_sentiment.update_layout(
            title='Sentiment Score Over Time',  # Title of the sentiment plot
            xaxis_title='Date',  # X-axis label
            yaxis_title='Sentiment Score',  # Y-axis label
            template="plotly_white",  # Template for the plot's appearance
        )

        # Display sentiment plot
        st.plotly_chart(fig_sentiment, use_container_width=True)

except Exception as e:
    # Display an error message if fetching sentiment data fails
    st.error(f"Error retrieving data: {e}")

# --- News and Sentiment Display ---
st.subheader('Recent News and Sentiment')
try:
    # Fetch news data from MongoDB related to the selected stock
    news_from_db = db["news_data"].find({"symbol": stock_symbol})
    news_df = pd.DataFrame(list(news_from_db))  # Convert news data to a DataFrame
    if not news_df.empty:
        # Drop unnecessary columns and format date
        news_df = news_df.drop(columns=['_id', 'symbol'])
        news_df['date'] = pd.to_datetime(news_df['date']).dt.strftime("%B %d, %Y")  # Format date for display

        # Create an "Open" hyperlink column for news articles
        news_df['Open'] = news_df['link'].apply(lambda link: f"<a href='{link}' target='_blank'>Open</a>")
        news_df = news_df.drop(columns=['link'])  # Drop the link column

        # Display news in a styled table with "Open" hyperlinks
        st.write(news_df.style.format({"sentiment": "{:.2f}"})
                 .applymap(lambda x: "color: green" if (isinstance(x, float) and x > 0) or (isinstance(x, str) and x.startswith('green')) 
                           else "color: red" if (isinstance(x, float) and x < 0) or (isinstance(x, str) and x.startswith('red'))
                           else "", subset=['sentiment'])  # Color coding for sentiment
                 .to_html(escape=False, index=False),
                 unsafe_allow_html=True)  # Allow HTML rendering

    else:
        st.write("No recent news found for this stock.")  # Message if no news found
except Exception as e:
    # Display an error message if fetching/displaying news fails
    st.error(f"Error fetching/displaying news: {e}")

# --- Prediction Table ---
st.subheader('Predictions')

if model:
    try:
        # Prepare prediction data
        data_train = data[['Close']][0: int(len(data) * 0.80)]  # Training data (80% of the dataset)
        data_test = data[['Close']][int(len(data) * 0.80):]  # Testing data (20% of the dataset)

        # Initialize a scaler for data normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit the scaler on the training data
        data_train_scaled = scaler.fit_transform(data_train)  

        # Get the last 100 days of the training data for predictions
        past_100_days = data_train.tail(100)

        # Concatenate the last 100 days of training data with the test data
        data_test = pd.concat([past_100_days, data_test], ignore_index=True)
        
        # Scale the test data
        data_test_scaled = scaler.transform(data_test)  

        x_test = []  # Initialize lists for features and labels
        y_test = []

        # Create sequences of 100 days for the test data
        for i in range(100, data_test_scaled.shape[0]):
            x_test.append(data_test_scaled[i-100:i])  # Features
            y_test.append(data_test_scaled[i, 0])  # Actual labels

        x_test, y_test = np.array(x_test), np.array(y_test)  # Convert lists to NumPy arrays
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshape for LSTM input

        # Make predictions using the loaded model
        predictions = model.predict(x_test)

        # Inverse transform to get actual prices from scaled predictions
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Create a DataFrame for actual and predicted values
        prediction_df = pd.DataFrame({
            'Actual': y_test.flatten(),  # Flatten the array for display
            'Predicted': predictions.flatten()  # Flatten the predictions
        })

        # Create binary labels for confusion matrix
        actual_movement = np.where(np.diff(y_test.flatten()) > 0, 1, 0)  # 1 for price increase, 0 for decrease
        predicted_movement = np.where(np.diff(predictions.flatten()) > 0, 1, 0)  # Same for predictions

        # Align lengths for confusion matrix
        actual_movement = actual_movement[1:]  # Adjust to match lengths
        predicted_movement = predicted_movement[1:]

        # Generate confusion matrix for the predictions
        cm = confusion_matrix(actual_movement, predicted_movement)

        # Calculate performance metrics for the predictions
        report = classification_report(actual_movement, predicted_movement, output_dict=True)
        sensitivity = report['1']['recall']  # Sensitivity for upward movement
        specificity = report['0']['recall']  # Specificity for downward movement
        f1 = report['1']['f1-score']  # F1 score
        accuracy = accuracy_score(actual_movement, predicted_movement)  # Overall accuracy

        # --- Plot Predicted vs Actual Prices ---
        prediction_dates = data.index[int(len(data) * 0.80) + 100:]  # Dates for predictions
        fig_prediction = go.Figure()
        # Add actual prices to the plot
        fig_prediction.add_trace(go.Scatter(x=prediction_dates, y=y_test.flatten(), mode='lines', name='Actual Price'))
        # Add predicted prices to the plot
        fig_prediction.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='Predicted Price'))
        
        # Finalize the layout for the prediction plot
        fig_prediction.update_layout(
            title=f'{stock_symbol} Actual vs Predicted Prices',
            xaxis_title='Date',  # X-axis label
            yaxis_title='Price',  # Y-axis label
            template="plotly_white",  # Template for the plot's appearance
            height=600  # Height of the plot
        )

        # Display the prediction plot in the app
        st.plotly_chart(fig_prediction, use_container_width=True)

        # Visualize confusion matrix using Seaborn
        fig_cm, ax = plt.subplots()  # Create a subplot for confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])  # Heatmap for confusion matrix
        ax.set_xlabel('Predicted')  # X-axis label
        ax.set_ylabel('Actual')  # Y-axis label
        ax.set_title('Confusion Matrix')  # Title for the confusion matrix

        # Create two columns to display results
        col1, col2 = st.columns(2)
        with col1:
            st.write(prediction_df)  # Display actual vs predicted prices
        with col2:
            st.pyplot(fig_cm)  # Display the confusion matrix plot

        # Display performance metrics
        st.subheader('Performance Metrics')
        st.write(f"**Sensitivity (Recall for Up):** {sensitivity:.2f}")  # Display sensitivity
        st.write(f"**Specificity (Recall for Down):** {specificity:.2f}")  # Display specificity
        st.write(f"**F1 Score:** {f1:.2f}")  # Display F1 score
        st.write(f"**Accuracy:** {accuracy:.2f}")  # Display overall accuracy

    except Exception as e:
        # Display an error message if making or displaying predictions fails
        st.error(f"Error making or displaying predictions: {e}")
else:
    st.warning("No model loaded. Cannot make predictions.")  # Warning if no model is loaded
