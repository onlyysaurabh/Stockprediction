import yfinance as yf
from pymongo import MongoClient

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB connection string
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]  # Replace with your database name
commodities_collection = db["commodities_data"]  # Collection to store commodity prices

def fetch_and_store_commodity_prices(symbol, name):
    """Fetches maximum available historical commodity prices and stores them in MongoDB."""
    try:
        data = yf.download(symbol, period="max")  # Fetch maximum available data

        # Format data for MongoDB
        formatted_data = []
        for index, row in data.iterrows():
            formatted_data.append({
                "Symbol": symbol,
                "Name": name,
                "Date": index.strftime("%Y-%m-%dT%H:%M:%S.000+00:00"),
                "Open": row['Open'],
                "High": row['High'],
                "Low": row['Low'],
                "Close": row['Close'],
                "Volume": row['Volume']
            })

        # Store in MongoDB
        commodities_collection.insert_one({"Commodity": name, "Data": formatted_data})
        print(f"Historical prices for {name} stored in MongoDB.")

    except Exception as e:
        print(f"Error fetching/storing data for {name}: {e}")


if __name__ == "__main__":
    # Fetch and store oil prices
    fetch_and_store_commodity_prices("CL=F", "Oil (WTI Crude)")

    # Fetch and store gold prices
    fetch_and_store_commodity_prices("GC=F", "Gold")

    # Fetch and store Bitcoin prices
    fetch_and_store_commodity_prices("BTC-USD", "Bitcoin")