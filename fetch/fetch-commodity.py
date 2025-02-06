import yfinance as yf
import pandas as pd
import datetime
from pymongo import MongoClient

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"  # Make sure MongoDB is running
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]
commodities_collection = db["commodities_data"]  # Use a more general name

def fetch_and_store_data(symbol, name):
    """
    Fetches historical data, formats it, and stores it in MongoDB.
    """
    try:
        data = yf.download(symbol, period="max")

        # Ensure data is not empty
        if data.empty:
            print(f"No data fetched for {name}. Check symbol: {symbol}")
            return

        print(data.head())  # Display first few rows for debugging

        # Format data for MongoDB
        formatted_data = []
        for index, row in data.iterrows():
            date_str = index.to_pydatetime().strftime("%Y-%m-%dT%H:%M:%S.000+00:00").split("T")[0]

            # Create the document to insert.  Use .values[0] to get the scalar value.
            document = {
                "Symbol": symbol,
                "Name": name,
                "Date": date_str,
                "Open": float(row["Open"].item()),
                "High": float(row["High"].item()),
                "Low": float(row["Low"].item()),
                "Close": float(row["Close"].item()),
            }
            # Handle adjusted close if it exists
            if "Adj Close" in row:
                document["Adj Close"] = float(row["Adj Close"].item())

            # Volume might not exist for all indices. Handle gracefully.
            if "Volume" in row:
                document["Volume"] = int(row["Volume"].item())
            else:
                document["Volume"] = 0  # Or None, depending on preference.

            formatted_data.append(document)

        # Insert data into MongoDB.
        if formatted_data:
            commodities_collection.insert_many(formatted_data)
            print(f"Stored {len(formatted_data)} records for {name} in MongoDB.")
        else:
            print(f"No data to insert for {name}.")


    except Exception as e:
        print(f"Error fetching/storing data for {name} ({symbol}): {e}")


if __name__ == "__main__":
    fetch_and_store_data("CL=F", "Oil (WTI Crude)")
    fetch_and_store_data("GC=F", "Gold")
    fetch_and_store_data("BTC-USD", "Bitcoin")
    fetch_and_store_data("^GSPC", "S&P 500")
    fetch_and_store_data("DX-Y.NYB", "US Dollar Index")