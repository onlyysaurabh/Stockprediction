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
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]
stock_data_collection = db["stock_data"]
stock_info_collection = db["stock_info"]
commodities_collection = db["commodities_data"]

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
col3, col4, col5, col6, col7 = st.columns(5)
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

# --- Sentiment Graph ---
st.subheader('Sentiment Over Time')
try:
    news_from_db = db["news_data"].find({"symbol": stock_symbol})
    news_df = pd.DataFrame(list(news_from_db))
    if not news_df.empty:
        news_df = news_df.drop(columns=['_id', 'symbol'])
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_df = news_df.sort_values(by='date')

        # Create sentiment plot
        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(go.Scatter(x=news_df['date'], y=news_df['sentiment'], mode='lines', name='Sentiment Score'))
        fig_sentiment.update_layout(
            title='Sentiment Score Over Time',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            template="plotly_white",
        )

        # Display sentiment plot
        st.plotly_chart(fig_sentiment, use_container_width=True)

except Exception as e:
    st.error(f"Error retrieving data: {e}")

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
        news_df = news_df.drop(columns=['link'])

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

# --- Prediction Table ---
st.subheader('Predictions')

if model:
    try:
        # Prepare prediction data
        data_train = data[['Close']][0: int(len(data) * 0.80)]
        data_test = data[['Close']][int(len(data) * 0.80):]

        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit the scaler on the training data
        data_train_scaled = scaler.fit_transform(data_train)  

        # Get the last 100 days of the training data
        past_100_days = data_train.tail(100)

        # Concatenate the last 100 days of training data with the test data
        data_test = pd.concat([past_100_days, data_test], ignore_index=True)
        
        # Scale the test data
        data_test_scaled = scaler.transform(data_test)  

        x_test = []
        y_test = []

        # Create sequences of 100 days for the test data
        for i in range(100, data_test_scaled.shape[0]):
            x_test.append(data_test_scaled[i-100:i])
            y_test.append(data_test_scaled[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshape for LSTM

        # Make predictions
        predictions = model.predict(x_test)

        # Inverse transform to get actual prices
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Create a DataFrame for actual and predicted values
        prediction_df = pd.DataFrame({
            'Actual': y_test.flatten(),
            'Predicted': predictions.flatten()
        })

        # Create binary labels for confusion matrix
        actual_movement = np.where(np.diff(y_test.flatten()) > 0, 1, 0)  # 1 for up, 0 for down
        predicted_movement = np.where(np.diff(predictions.flatten()) > 0, 1, 0)

        # Align lengths for confusion matrix
        actual_movement = actual_movement[1:]  # Adjust to match the length
        predicted_movement = predicted_movement[1:]

        # Generate confusion matrix
        cm = confusion_matrix(actual_movement, predicted_movement)

        # Calculate performance metrics
        report = classification_report(actual_movement, predicted_movement, output_dict=True)
        sensitivity = report['1']['recall']
        specificity = report['0']['recall']
        f1 = report['1']['f1-score']
        accuracy = accuracy_score(actual_movement, predicted_movement)

        # --- Plot Predicted vs Actual Prices ---
        prediction_dates = data.index[int(len(data) * 0.80) + 100:]  # Corresponding dates for predictions
        fig_prediction = go.Figure()
        fig_prediction.add_trace(go.Scatter(x=prediction_dates, y=y_test.flatten(), mode='lines', name='Actual Price'))
        fig_prediction.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='Predicted Price'))
        
        fig_prediction.update_layout(
            title=f'{stock_symbol} Actual vs Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Price',
            template="plotly_white",
            height=600
        )

        st.plotly_chart(fig_prediction, use_container_width=True)

        # Visualize confusion matrix using Seaborn
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')

        # Display the confusion matrix and metrics in Streamlit
        col1, col2 = st.columns(2)
        with col1:
            st.write(prediction_df)

        with col2:
            st.pyplot(fig_cm)

        # Display performance metrics
        st.subheader('Performance Metrics')
        st.write(f"**Sensitivity (Recall for Up):** {sensitivity:.2f}")
        st.write(f"**Specificity (Recall for Down):** {specificity:.2f}")
        st.write(f"**F1 Score:** {f1:.2f}")
        st.write(f"**Accuracy:** {accuracy:.2f}")

    except Exception as e:
        st.error(f"Error making or displaying predictions: {e}")
else:
    st.warning("No model loaded. Cannot make predictions.")
