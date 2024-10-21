import datetime
import json
import time
from GoogleNews import GoogleNews
from pymongo import MongoClient
from transformers import pipeline
from dateutil.relativedelta import relativedelta
import pandas as pd

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB connection string
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]  # Replace with your database name
news_data_collection = db["news_data"]

# --- FinBERT Pipeline ---
classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')

# --- JSON file for tracking progress ---
PROGRESS_FILE = "news_fetch_progress.json"

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
            date = parse_date(date_str)
            formatted_date = date.strftime("%Y-%m-%dT%H:%M:%S.000+00:00") if date else None

            # Get sentiment score using FinBERT
            sentiment = get_sentiment(title)

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

def parse_date(date_str):
    """Parses the date string into a datetime object."""
    try:
        parts = date_str.split()

        if len(parts) == 3 and parts[1] in ('hour', 'hours', 'minute', 'minutes', 'day', 'days', 'week', 'weeks', 'month', 'months'):
            num_units = int(parts[0])
            unit = parts[1].rstrip('s')  # Remove 's' from 'hours', 'minutes', 'days', 'weeks', 'months'

            if unit in ('hour', 'hours'):
                date = datetime.datetime.now() - relativedelta(hours=num_units)
            elif unit in ('minute', 'minutes'):
                date = datetime.datetime.now() - relativedelta(minutes=num_units)
            elif unit in ('day', 'days'):
                date = datetime.datetime.now() - relativedelta(days=num_units)
            elif unit in ('week', 'weeks'):
                date = datetime.datetime.now() - relativedelta(weeks=num_units)
            elif unit in ('month', 'months'):
                date = datetime.datetime.now() - relativedelta(months=num_units)
            else:
                # If the relative date format is not recognized, set date to None
                date = None
        else:
            # Handle other date formats (e.g., "Oct 18, 2024")
            date = datetime.datetime.strptime(date_str, "%b %d, %Y")

        return date  # Return the datetime object directly

    except ValueError:
        print(f"Error parsing date: {date_str}")
        return None

def get_sentiment(title):
    """Gets the sentiment score for the given title using FinBERT."""
    try:
        result = classifier(title)[0]
        return result['score']
    except Exception as e:
        print(f"Error getting sentiment for '{title}': {e}")
        return 0  # Assign a neutral sentiment if there's an error

def load_progress():
    """Loads progress from the JSON file."""
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}  # Return an empty dictionary if the file doesn't exist

def save_progress(progress):
    """Saves progress to the JSON file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)

if __name__ == "__main__":
    try:
        stocks_df = pd.read_csv('stocks.csv')  # Read stock symbols and names from CSV
    except Exception as e:
        print(f"Error reading stocks.csv: {e}")
        exit()

    # Specify the date range in mm/dd/yyyy format (for news data)
    start_date = "10/01/2024"  
    end_date = "10/21/2024"

    progress = load_progress()  # Load progress from JSON

    # Find the index of the last processed stock (if any)
    last_processed_index = 0
    for i, row in stocks_df.iterrows():
        if row['Symbol'] in progress:
            last_processed_index = i

    # Iterate from the last processed stock
    for index, row in stocks_df.iloc[last_processed_index:].iterrows():
        stock_symbol = row['Symbol']
        stock_name = row['Name']

        fetch_and_store_news(stock_symbol, stock_name, start_date, end_date)
        progress[stock_symbol] = True  # Mark this stock as done
        save_progress(progress)  # Save progress to JSON

        # Add a sleep timer to prevent IP blocking
        time.sleep(5)  # Sleep for 5 seconds (adjust as needed)