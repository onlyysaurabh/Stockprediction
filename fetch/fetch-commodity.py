import yfinance as yf
import pandas as pd
import datetime
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"  # Or your MongoDB URI
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]  # Or your desired database name
commodities_collection = db["commodities_data"]  # Or your desired collection name


def fetch_and_store_commodity_prices(symbol, name):
    """
    Fetches historical commodity prices from Yahoo Finance and stores them in MongoDB.

    Args:
        symbol (str): The Yahoo Finance ticker symbol for the commodity.
        name (str): The name of the commodity.
    """
    try:
        # Download data with period="max" to get all available historical data
        data = yf.download(symbol, period="max")

        # Check if data was retrieved. yfinance sometimes returns an empty DataFrame.
        if data.empty:
            print(f"No data fetched for {name}. Check symbol: {symbol}")
            return  # Exit the function if no data is found


        # Convert data to a format suitable for MongoDB and handle potential errors
        formatted_data = []
        for index, row in data.iterrows():
            # Format the date to ISO 8601 format (YYYY-MM-DD)
            date = index.to_pydatetime().strftime("%Y-%m-%d")

            # Create a document for each day's data
            try:
                # Convert to appropriate types.  Use .get() with a default value to handle potential missing columns.
                record = {
                    "Symbol": symbol,
                    "Name": name,
                    "Date": date,
                    "Open": float(row.get("Open", None)),  # Handle if "Open" is missing
                    "High": float(row.get("High", None)),  # Handle if "High" is missing
                    "Low": float(row.get("Low", None)),    # Handle if "Low" is missing
                    "Close": float(row.get("Close", None)), # Handle if "Close" is missing
                    "Volume": int(row.get("Volume", 0)) if not pd.isna(row.get("Volume", None)) else 0  # Handle missing and NaN volumes. Convert to int.

                }
                # Handle cases where Adj Close might exist:
                if "Adj Close" in row:
                    record["Adj Close"] = float(row["Adj Close"])
                formatted_data.append(record)

            except (TypeError, ValueError) as e:
                print(f"Error converting data for {name} on {date}: {e}. Skipping this row.")
                continue  # Skip this row and continue with the next


        # Store the data in MongoDB
        if formatted_data:  # Only insert if there's valid data
            # Using insert_many for better performance with large datasets.
            commodities_collection.insert_many(formatted_data)
            print(f"Stored {len(formatted_data)} records for {name}.")
        else:
            print(f"No valid data to insert for {name}.")


    except Exception as e:
        print(f"Error fetching/storing data for {name}: {e}")



if __name__ == "__main__":
    # Commodity and index data to fetch
    commodities = [
        ("CL=F", "Oil (WTI Crude)"),
        ("GC=F", "Gold"),
        ("HG=F", "Copper"),        # Added Copper
        ("NG=F", "Natural Gas"),  # Added Natural Gas
        ("BTC-USD", "Bitcoin"),
        ("^GSPC", "S&P 500"),
        ("DX=F", "US Dollar Index"),
        ("^IXIC", "NASDAQ Composite")              # Added NASDAQ
    ]

    for symbol, name in commodities:
        fetch_and_store_commodity_prices(symbol, name)

    client.close() #close the connection after
    print("Data fetching and storing complete.")