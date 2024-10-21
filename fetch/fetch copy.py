import datetime
from GoogleNews import GoogleNews
from pymongo import MongoClient
from transformers import pipeline
from dateutil.relativedelta import relativedelta
import yfinance as yf
import pandas as pd

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB connection string
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]  # Replace with your database name
stock_data_collection = db["stock_data"]
news_data_collection = db["news_data"]

# --- FinBERT Pipeline ---
classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')

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


def fetch_and_store_news(stock_symbol, stock_name, start_date, end_date):
    """
    Fetches news from Google News within a date range, 
    analyzes sentiment using FinBERT, and stores the data in MongoDB.
    """
    try:
        googlenews = GoogleNews(lang='en')
        googlenews.set_time_range(start_date, end_date)  # Set the date range
        
        search_query = f"{stock_symbol} {stock_name}"
        googlenews.search(search_query)
        news_results = googlenews.results()

        news_data = []
        for news in news_results:
            title = news['title']
            date_str = news['date']
            link = news['link']

            # Parse the date string and convert to datetime object
            try:
                parts = date_str.split()

                if len(parts) == 3 and parts[1] in ('hour', 'hours', 'minute', 'minutes', 'day', 'days'):
                    num_units = int(parts[0])
                    unit = parts[1].rstrip('s') 

                    if unit in ('hour', 'hours'):
                        date = datetime.datetime.now() - relativedelta(hours=num_units)
                    elif unit in ('minute', 'minutes'):
                        date = datetime.datetime.now() - relativedelta(minutes=num_units)
                    elif unit in ('day', 'days'):
                        date = datetime.datetime.now() - relativedelta(days=num_units)
                    else:
                        # If the relative date format is not recognized, set date to None
                        date = None
                else:
                    # Handle other date formats (e.g., "Oct 18, 2024")
                    date = datetime.datetime.strptime(date_str, "%b %d, %Y")
                
                # Format the date as "1962-01-02T05:00:00.000+00:00"
                formatted_date = date.strftime("%Y-%m-%dT%H:%M:%S.000+00:00") if date else None

            except ValueError:  # Indentation fixed here
                print(f"Error parsing date: {date_str}")
                formatted_date = None

            # Get sentiment score using FinBERT
            try:
                result = classifier(title)[0]
                sentiment = result['score']
            except Exception as e:
                print(f"Error getting sentiment for '{title}': {e}")
                sentiment = 0  # Assign a neutral sentiment if there's an error

            news_data.append({
                "symbol": stock_symbol,
                "date": formatted_date,  # Store the formatted date
                "title": title,
                "link": link,
                "sentiment": sentiment
            })

        # Store the news data in MongoDB
        if news_data:
            news_data_collection.insert_many(news_data)
            print(f"News data for {stock_symbol} stored in MongoDB.")
        else:
            print(f"No news found for {stock_symbol}.")

    except Exception as e:
        print(f"Error fetching/analyzing news for {stock_symbol}: {e}")


if __name__ == "__main__":
    try:
        stocks_df = pd.read_csv('stocks.csv')  # Read stock symbols and names from CSV
    except Exception as e:
        print(f"Error reading stocks.csv: {e}")
        exit()

    # Specify the date range in mm/dd/yyyy format (for news data)
    start_date = "10/01/2024"  
    end_date = "10/20/2024"

    for index, row in stocks_df.iterrows():
        stock_symbol = row['Symbol']
        stock_name = row['Name']
        
        fetch_and_store_stock_data(stock_symbol)
        fetch_and_store_news(stock_symbol, stock_name, start_date, end_date)