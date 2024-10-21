import yfinance as yf
from pymongo import MongoClient
import pandas as pd

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB connection string
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]  # Replace with your database name
stock_info_collection = db["stock_info"]  # Collection to store stock info

def fetch_and_store_stock_info(stock_symbol):
    """Fetches stock information from yfinance and stores it in MongoDB."""
    try:
        stock = yf.Ticker(stock_symbol)

        # Get the relevant information you need
        info_dict = {
            "symbol": stock_symbol,
            "companyName": stock.info['longName'],
            "sector": stock.info.get('sector'),
            "industry": stock.info.get('industry'),
            "marketCap": stock.info.get('marketCap'),
            "trailingPE": stock.info.get('trailingPE'),
            "forwardPE": stock.info.get('forwardPE'),
            "dividendYield": stock.info.get('dividendYield'),
            # Add other info fields as needed...
        }

        # Store in MongoDB
        stock_info_collection.insert_one(info_dict)
        print(f"Stock info for {stock_symbol} stored in MongoDB.")

    except Exception as e:
        print(f"Error fetching/storing stock info for {stock_symbol}: {e}")


if __name__ == "__main__":
    try:
        stocks_df = pd.read_csv('stocks.csv')  # Read stock symbols from CSV
    except Exception as e:
        print(f"Error reading stocks.csv: {e}")
        exit()

    for index, row in stocks_df.iterrows():
        stock_symbol = row['Symbol']
        fetch_and_store_stock_info(stock_symbol)