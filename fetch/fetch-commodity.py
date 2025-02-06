import yfinance as yf
import pandas as pd
import datetime
from pymongo import MongoClient

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]
commodities_collection = db["commodities_data"]

def fetch_and_store_commodity_prices(symbol, name):
    """Fetches historical commodity prices and stores them in MongoDB."""
    try:
        data = yf.download(symbol, period="max")

        # Ensure data is not empty
        if data.empty:
            print(f"No data fetched for {name}. Check symbol: {symbol}")
            return
        
        print(data.head())  # Display first few rows

        # Format data for MongoDB
        formatted_data = []
        for index, row in data.iterrows():
            i=0
            date = index.to_pydatetime().strftime("%Y-%m-%dT%H:%M:%S.000+00:00").split("T")[0]
            # print(date)
            # print(type(index.to_pydatetime()), index.to_pydatetime())
            # print(type(row["Open"]), float(row["Open"].iloc[i]))
            # print(type(row["High"]), float(row["High"].iloc[i]))
            # print(type(row["Low"]), float(row["Low"].iloc[i]))
            # print(type(row["Close"]), float(row["Close"].iloc[i]))
            # print(type(row["Volume"]), float(row["Volume"].iloc[i]))

            formatted_data.append({
                "Symbol": symbol,
                "Name": name,
                "Date": date,
                "Open": float(row["Open"].iloc[i]),
                "High": float(row["High"].iloc[i]),
                "Low": float(row["Low"].iloc[i]),
                "Close": float(row["Close"].iloc[i]),
                "Volume": int(row["Volume"].iloc[i])
            })
            i+=1
        
        print(len(formatted_data))

        # Insert data in MongoDB
        if len(formatted_data) > 0:
            commodities_collection.insert_many(formatted_data)
            print(f"Stored {len(formatted_data)} records for {name} in MongoDB.")
        else:
            print(f"No valid data to insert for {name}.")

    except Exception as e:
        print(f"Error fetching/storing data for {name}: {e}")

if __name__ == "__main__":
    fetch_and_store_commodity_prices("CL=F", "Oil (WTI Crude)")
    fetch_and_store_commodity_prices("GC=F", "Gold")
    fetch_and_store_commodity_prices("BTC-USD", "Bitcoin")
