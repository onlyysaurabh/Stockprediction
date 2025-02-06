import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pymongo import MongoClient
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score  # Regression metrics
import pickle  # For loading scikit-learn models

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]
stock_data_collection = db["stock_data"]
stock_info_collection = db["stock_info"]
commodities_collection = db["commodities_data"]

st.set_page_config(page_title="Stock Market App", layout="wide", initial_sidebar_state="collapsed")  # Hide sidebar

# --- App Styling ---
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



# Ensure user is logged in
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.warning("You are not logged in. Redirecting to login page...")
    st.switch_page("index.py")  # Redirect to login

# Custom Logout Button at the top right
st.markdown(
    """<style>
        .logout-button {
            position: fixed;
            top: 10px;
            right: 20px;
            background-color: red;
            color: white;
            padding: 8px 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            z-index: 9999;
        }
        .logout-button:hover {
            background-color: darkred;
        }
    </style>""",
    unsafe_allow_html=True
)

# Display the Logout Button
if st.button("Logout", key="logout", help="Logout from the app"):
    st.session_state.clear()  # Clears all session data
    st.session_state["logged_in"] = False
    st.success("Logged out successfully! Redirecting...")
    st.switch_page("index.py")

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


# --- Stock Information ---
with col2:
    st.subheader('Stock Information')
    stock_info_from_db = stock_info_collection.find_one({"symbol": stock_symbol})
    if stock_info_from_db:
        st.success("Stock info loaded from MongoDB!")  # Success message
        # Convert stock info to a DataFrame
        info_df = pd.DataFrame(list(stock_info_from_db.items()), columns=['Attribute', 'Value'])
        info_df = info_df[info_df['Attribute'] != '_id']  # Exclude MongoDB's internal ID field
        st.table(info_df)
    else:
        st.error(f"No stock information found for {stock_symbol} in MongoDB.")

# --- Load Models ---
model_paths = {
    'lstm': f'./data/{stock_symbol}/Stock_Predictions_Model.keras',
    'svm': f'./data/{stock_symbol}/svm_model.pkl',
    'rf': f'./data/{stock_symbol}/rf_model.pkl',
    'xgb': f'./data/{stock_symbol}/xgb_model.pkl',
}
models = {}
for model_name, model_path in model_paths.items():
    if os.path.exists(model_path):
        if model_name == 'lstm':
            models[model_name] = load_model(model_path)
        else:
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
        st.success(f"{model_name.upper()} model loaded successfully!")
    else:
        st.warning(f"No saved {model_name.upper()} model found for {stock_symbol}.")


# --- Historical Data Chart (as before, unchanged)---

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

# --- Predictions ---
st.subheader('Predictions')

if models:
    try:
        # Prepare data (same as before, but adapted for all models)
        data_close = data[['Close']]
        train_data_len = int(np.ceil(len(data_close) * 0.8))
        train_data = data_close[0: train_data_len]
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data_scaled = scaler.fit_transform(train_data)
        past_100_days = train_data.tail(100)
        test_data = data_close[train_data_len - 100:] # Corrected line
        test_data_scaled = scaler.transform(test_data)

        x_test = []
        y_test = data_close[train_data_len:].values  # Actual values for comparison

        for i in range(100, len(test_data_scaled)):
            x_test.append(test_data_scaled[i-100:i, 0])
        x_test = np.array(x_test)

        # Store predictions for each model
        predictions = {}

        # LSTM Predictions
        if 'lstm' in models:
            x_test_lstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            predictions['lstm'] = models['lstm'].predict(x_test_lstm)
            predictions['lstm'] = scaler.inverse_transform(predictions['lstm'])

        # Other Models (SVM, RF, XGB) - No reshaping needed
        for model_name in ['svm', 'rf', 'xgb']:
            if model_name in models:
                predictions[model_name] = models[model_name].predict(x_test)
                # No inverse transform needed if the model predicts directly on scaled data

        # Plot all predictions
        fig_prediction = go.Figure()
        fig_prediction.add_trace(go.Scatter(x=data_close.index[train_data_len:], y=y_test.flatten(), mode='lines', name='Actual Price'))

        for model_name, pred in predictions.items():
            # If LSTM, use the flattened prediction.  Otherwise, use pred directly
            y_values = pred.flatten() if model_name == 'lstm' else pred
            fig_prediction.add_trace(go.Scatter(x=data_close.index[train_data_len:], y=y_values, mode='lines', name=f'Predicted Price ({model_name.upper()})'))



        fig_prediction.update_layout(
            title=f'{stock_symbol} Actual vs Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Price',
            template="plotly_white",
            height=600
        )
        st.plotly_chart(fig_prediction, use_container_width=True)


        # --- Evaluation Table ---
        st.subheader('Model Evaluation')
        evaluation_data = []

        for model_name, pred in predictions.items():
            # If LSTM, use the flattened prediction.  Otherwise, use pred directly
            y_pred_values = pred.flatten() if model_name == 'lstm' else pred

            rmse = np.sqrt(mean_squared_error(y_test, y_pred_values))
            r2 = r2_score(y_test, y_pred_values)
            evaluation_data.append([model_name.upper(), rmse, r2])

        evaluation_df = pd.DataFrame(evaluation_data, columns=['Model', 'RMSE', 'R-squared'])
        st.table(evaluation_df)


    except Exception as e:
        st.error(f"Error making or displaying predictions: {e}")
else:
    st.warning("No models loaded. Cannot make predictions.")
