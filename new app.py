import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pymongo import MongoClient
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import google.generativeai as genai  # Import the Gemini library

# --- Gemini API Setup ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # Directly use from environment
if not GOOGLE_API_KEY:
    st.error("Google API Key not found!  Set the GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')


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


# --- Stock Information ---
stock_info = {}  # Initialize as an empty dictionary
with col2:
    st.subheader('Stock Information')
    stock_info_from_db = stock_info_collection.find_one({"symbol": stock_symbol})
    if stock_info_from_db:
      st.success("Stock info loaded from MongoDB!")
      info_df = pd.DataFrame(list(stock_info_from_db.items()), columns=['Attribute', 'Value'])
      info_df = info_df[info_df['Attribute'] != '_id']
      st.table(info_df)
      # Convert the stock info to a dictionary for Gemini
      stock_info = dict(zip(info_df['Attribute'], info_df['Value']))
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



# --- Historical Data Chart ---
st.subheader('Select Parameters for Historical Data')

all_commodities = [
    ("CL=F", "Oil (WTI Crude)"),
    ("GC=F", "Gold"),
    ("HG=F", "Copper"),
    ("NG=F", "Natural Gas"),
    ("BTC-USD", "Bitcoin"),
    ("^GSPC", "S&P 500"),
    ("DX=F", "US Dollar Index"),
    ("^IXIC", "NASDAQ Composite")
]

num_cols = 4
cols = st.columns(num_cols)
selected_commodities = []

with cols[0]:
    show_close = st.checkbox('Closing Price', value=True)
    show_ma50 = st.checkbox('MA50', value=True)

for i, (symbol, name) in enumerate(all_commodities):
    col_index = (i + 1) % num_cols
    with cols[col_index]:
        if st.checkbox(name, value=False):
            selected_commodities.append((symbol, name))

fig = go.Figure()

if show_close:
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price'))
if show_ma50:
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(50).mean(), mode='lines', name='MA50'))

for symbol, commodity_name in selected_commodities:
    commodity_data_cursor = commodities_collection.find({"Name": commodity_name})
    commodity_data_list = list(commodity_data_cursor)
    if commodity_data_list:
        commodity_df = pd.DataFrame()
        for doc in commodity_data_list:
            temp_df = pd.DataFrame(doc, index=[0])
            commodity_df = pd.concat([commodity_df, temp_df], ignore_index=True)
        commodity_df['Date'] = pd.to_datetime(commodity_df['Date'])
        commodity_df.set_index('Date', inplace=True)
        commodity_df.sort_index(inplace=True)
        fig.add_trace(go.Scatter(x=commodity_df.index, y=commodity_df['Close'], mode='lines', name=commodity_name))
    else:
        st.warning(f"No data found for {commodity_name} in MongoDB.")

fig.update_layout(
    title=f'{stock_symbol} Stock Price',
    xaxis_title='Time',
    yaxis_title='Price',
    height=600,
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)


# --- Sentiment Graph and News ---
st.subheader('Sentiment Over Time')
news_df = pd.DataFrame() # initialize empty dataframe
try:
    news_from_db = db["news_data"].find({"symbol": stock_symbol})
    news_df = pd.DataFrame(list(news_from_db))
    if not news_df.empty:
        news_df_display = news_df.drop(columns=['_id', 'symbol']) # Create a copy for display
        news_df_display['date'] = pd.to_datetime(news_df_display['date']).dt.strftime("%B %d, %Y")
        news_df_display['Open'] = news_df_display['link'].apply(lambda link: f"<a href='{link}' target='_blank'>Open</a>")
        news_df_display = news_df_display.drop(columns=['link'])
        st.write(news_df_display.style.format({"sentiment": "{:.2f}"})
                 .applymap(lambda x: "color: green" if (isinstance(x, float) and x > 0) or (isinstance(x, str) and x.startswith('green'))
                           else "color: red" if (isinstance(x, float) and x < 0) or (isinstance(x, str) and x.startswith('red'))
                           else "", subset=['sentiment'])
                 .to_html(escape=False, index=False),
                 unsafe_allow_html=True)

        # Sentiment Plot (using the original news_df for calculations)
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_df = news_df.sort_values(by='date')
        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(go.Scatter(x=news_df['date'], y=news_df['sentiment'], mode='lines', name='Sentiment Score'))
        fig_sentiment.update_layout(title='Sentiment Score Over Time', xaxis_title='Date', yaxis_title='Sentiment Score', template="plotly_white")
        st.plotly_chart(fig_sentiment, use_container_width=True)

    else:
        st.write("No recent news found for this stock.")
except Exception as e:
    st.error(f"Error retrieving news/sentiment data: {e}")


# --- Predictions ---
st.subheader('Predictions')
predictions_data = {} # Store predictions for summarization
if models:
    try:
        data_close = data[['Close']]
        train_data_len = int(np.ceil(len(data_close) * 0.8))
        train_data = data_close[0: train_data_len]
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data_scaled = scaler.fit_transform(train_data)
        past_100_days = train_data.tail(100)
        test_data = data_close[train_data_len - 100:]
        test_data_scaled = scaler.transform(test_data)

        x_test = []
        y_test = data_close[train_data_len:].values

        for i in range(100, len(test_data_scaled)):
            x_test.append(test_data_scaled[i-100:i, 0])
        x_test = np.array(x_test)

        predictions = {}

        if 'lstm' in models:
          x_test_lstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
          predictions['lstm'] = models['lstm'].predict(x_test_lstm)
          predictions['lstm'] = scaler.inverse_transform(predictions['lstm'])

        for model_name in ['svm', 'rf', 'xgb']:
            if model_name in models:
                predictions[model_name] = models[model_name].predict(x_test)

        fig_prediction = go.Figure()
        fig_prediction.add_trace(go.Scatter(x=data_close.index[train_data_len:], y=y_test.flatten(), mode='lines', name='Actual Price'))

        for model_name, pred in predictions.items():
            y_values = pred.flatten() if model_name == 'lstm' else pred
            fig_prediction.add_trace(go.Scatter(x=data_close.index[train_data_len:], y=y_values, mode='lines', name=f'Predicted Price ({model_name.upper()})'))
            predictions_data[model_name] = y_values  # Store predictions


        fig_prediction.update_layout(
            title=f'{stock_symbol} Actual vs Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Price',
            template="plotly_white",
            height=600
        )
        st.plotly_chart(fig_prediction, use_container_width=True)

        st.subheader('Model Evaluation')
        evaluation_data = []

        for model_name, pred in predictions.items():
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



# --- Gemini Summarization ---
st.subheader("Summary and Future Scope")

if st.button("Generate Summary"):
    if not stock_info:  # Check if stock_info is empty
        st.error("Cannot generate summary: No stock information available.")
        st.stop()
    with st.spinner("Generating summary..."):
        try:
            # Prepare the prompt for Gemini
            prompt = f"""
            Summarize the following information about the stock {stock_symbol} and provide insights into its future scope.  Include historical performance, recent news sentiment, and model predictions.

            Stock Information:
            {stock_info}

            Historical Data Summary:
            The stock's closing prices and 50-day moving average are available.  (Note: We don't provide the raw historical data to Gemini, just a description).

            Recent News and Sentiment:
            {news_df.to_string() if not news_df.empty else 'No recent news available.'}

            Model Predictions (Closing Price):
            """
            # Add predictions to the prompt, handling potential shape differences
            for model_name, pred_values in predictions_data.items():
                prompt += f"- {model_name.upper()}: {pred_values.tolist()}\n" # Convert to list

            prompt += "\nProvide a concise summary and outlook for the stock."

            # Get the summary from Gemini
            response = model.generate_content(prompt)
            st.write(response.text)

        except Exception as e:
            st.error(f"Error generating summary: {e}")