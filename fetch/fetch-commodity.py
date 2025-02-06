import yfinance as yf
import pandas as pd
import datetime
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]
commodities_collection = db["commodities_data"]

def fetch_and_store_commodity_prices(symbol, name):
    try:
        data = yf.download(symbol, period="max")
        if data.empty:
            print(f"No data fetched for {name}. Check symbol: {symbol}")
            return

        formatted_data = []
        for index, row in data.iterrows():
            date = index.to_pydatetime().strftime("%Y-%m-%dT%H:%M:%S.000+00:00").split("T")[0]
            formatted_data.append({
                "Symbol": symbol,
                "Name": name,
                "Date": date,
                "Open": float(row["Open"]),
                "High": float(row["High"]),
                "Low": float(row["Low"]),
                "Close": float(row["Close"]),
                "Volume": int(row["Volume"])
            })

        if formatted_data:
            commodities_collection.insert_many(formatted_data)
            print(f"Stored {len(formatted_data)} records for {name}.")
        else:
            print(f"No valid data to insert for {name}.")
    except Exception as e:
        print(f"Error fetching/storing data for {name}: {e}")

if __name__ == "__main__":
    fetch_and_store_commodity_prices("CL=F", "Oil (WTI Crude)")
    fetch_and_store_commodity_prices("GC=F", "Gold")
    fetch_and_store_commodity_prices("BTC-USD", "Bitcoin")
    fetch_and_store_commodity_prices("^GSPC", "S&P 500")
    fetch_and_store_commodity_prices("DX=F", "US Dollar Index")
