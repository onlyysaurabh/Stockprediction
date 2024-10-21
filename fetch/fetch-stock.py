import yfinance as yf
from pymongo import MongoClient
import pandas as pd

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB connection string
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]  # Replace with your database name
stock_data_collection = db["stock_data"]

def fetch_and_store_stock_data(stock_symbol):
    """Fetches stock data from yfinance and stores it in MongoDB."""
    try:
        stock = yf.Ticker(stock_symbol)
        data = stock.history(period="max")
        
        # Format data for MongoDB
        formatted_data = []
        for index, row in data.iterrows():
            formatted_data.append({
                "Symbol": stock_symbol,
                "Date": index.strftime("%Y-%m-%dT%H:%M:%S.000+00:00"),
                "Open": row['Open'],
                "High": row['High'],
                "Low": row['Low'],
                "Close": row['Close'],
                "Volume": row['Volume']
            })

        # Store in MongoDB
        stock_data_collection.insert_one({"Symbol": stock_symbol, "Data": formatted_data})
        print(f"Stock data for {stock_symbol} stored in MongoDB.")

    except Exception as e:
        print(f"Error fetching/storing stock data for {stock_symbol}: {e}")


if __name__ == "__main__":
    try:
        stocks_df = pd.read_csv('stocks.csv')  # Read stock symbols from CSV
    except Exception as e:
        print(f"Error reading stocks.csv: {e}")
        exit()

    for index, row in stocks_df.iterrows():
        stock_symbol = row['Symbol']
        fetch_and_store_stock_data(stock_symbol)