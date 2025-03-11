import json
import os
import logging
import sys
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from io import StringIO

log_filename = r"C:\Users\kause\OneDrive - The University of Texas at Dallas\algo_trader\presentation\code\logfile.log"
CONFIG_FILE = r"C:\Users\kause\OneDrive - The University of Texas at Dallas\algo_trader\presentation\code\config.json"

logging.basicConfig(
    level=logging.DEBUG,  # Log level can be DEBUG, INFO, WARNING, ERROR, or CRITICAL.
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_filename),  # Write log messages to a file.
        logging.StreamHandler(sys.stdout)   # Optionally also print to the console.
    ]
)

def load_config():
    """Load configuration from config.json."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"{CONFIG_FILE} not found.")
    return config

def save_config(config):
    """Save the updated configuration back to config.json."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)


def fetch_alpha_vantage_data(ticker, start_month, end_month, interval, API_KEY, BASE_URL, data_dir):
    """
    Fetches intraday stock data from Alpha Vantage for a given ticker and time range.

    Parameters:
    - ticker: Stock ticker symbol (e.g., "AAPL")
    - start_month: Start month in 'YYYY-MM' format
    - end_month: End month in 'YYYY-MM' format
    - interval: Time interval for intraday data (default: "5min")

    Returns:
    - Pandas DataFrame containing the fetched data, or None if no data is retrieved.

    Saves the fetched data to a CSV file.
    """

    # Convert start and end months to datetime objects
    start_date = datetime.strptime(start_month, "%Y-%m")
    end_date = datetime.strptime(end_month, "%Y-%m")

    # Create an empty list to store data chunks
    all_data = []

    # Iterate over each month in the range
    current_date = start_date
    while current_date <= end_date:
        year_month = current_date.strftime("%Y-%m")  # Format as YYYY-MM
        print(f"Fetching data for {ticker} - {year_month}...")

        # Construct API request URL
        url = f"{BASE_URL}?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={interval}&month={year_month}&outputsize=full&apikey={API_KEY}&datatype=csv"

        try:
            # Fetch data
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for failed requests

            # Convert CSV response to Pandas DataFrame
            df = pd.read_csv(StringIO(response.text)) # Use io.StringIO

            # Append data to list
            all_data.append(df)

            print(f"Data for {ticker} - {year_month} fetched successfully.")

        except Exception as e:
            print(f"Error fetching data for {ticker} - {year_month}: {e}")

        # Move to the next month
        current_date += timedelta(days=32)
        current_date = current_date.replace(day=1)

        # Avoid hitting API rate limits
        time.sleep(15)  # Adjust based on Alpha Vantage API rate limits

    # Combine all data chunks into a single DataFrame
    if all_data:
        final_data = pd.concat(all_data, ignore_index=True)

        # Save to CSV file
        file_name = f"{ticker}_intraday_data_{start_month}_to_{end_month}_{interval}.csv"
        file_path = os.path.join(data_dir, file_name)
        final_data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
        logging.info(f"Data saved to {file_path}")

        # Return the final DataFrame
        return final_data  # Return the DataFrame

    else:
        print(f"No data retrieved for {ticker}.")
        return None  # Return None if no data

def download_all_stock_data(ticker, start_month, end_month, API_KEY, BASE_URL, interval, data_dir):
    """
    Simulate downloading all historical stock data for a given ticker
    between start_month and end_month. Replace this function with your
    actual download logic.
    """
    print(f"Downloading data for {ticker} from {start_month} to {end_month}...")
    df = fetch_alpha_vantage_data(ticker=ticker, start_month=start_month, end_month=end_month, interval=interval, API_KEY=API_KEY, BASE_URL=BASE_URL, data_dir=data_dir) # Assign the returned DataFrame to df
    logging.info(f"Data downloaded for {ticker} from {start_month} to {end_month}.")
    logging.info(f"data dimensions: {df.shape}") 
    return True

def main():
    logging.info("Task started.")
    config = load_config()
    tickers = config.get("tickers", [])
    progress = config.get("progress", {})
    daily_download_limit = int(config.get("daily_download_limit", 1))
    downloaded = 0

    # Alpha Vantage API Key (Replace with your actual key)
    API_KEY = config.get("API_KEY", "demo")  # Replace with your API key
    BASE_URL = config.get("BASE_URL", "https://www.alphavantage.co/query")
    interval = config.get("interval", "1min")
    data_dir = config.get("data_dir", "")

    for ticker in tickers:
        ticker_progress = progress.get(ticker)
        if not ticker_progress:
            print(f"No progress configuration found for {ticker}. Skipping...")
            continue
        
        # Check if data has already been downloaded.
        if ticker_progress.get("status") == "downloaded":
            print(f"{ticker} data already downloaded. Skipping...")
            continue
        
        start_month = ticker_progress.get("start_month")
        end_month = ticker_progress.get("end_month")
        
        # Download all data at once for the ticker.
        if  downloaded < daily_download_limit:
            downloaded += 1
            if download_all_stock_data(ticker, start_month, end_month, API_KEY, BASE_URL, interval, data_dir):
                ticker_progress["status"] = "downloaded"
                print(f"{ticker}: Data downloaded successfully. Status updated to 'downloaded'.")
                logging.info(f"{ticker}: Data downloaded successfully. Status updated to 'downloaded'.")
            else:
                print(f"{ticker}: Download failed. Status remains pending.")
        else:
            print(f"Daily download limit reached. Skipping the rest of the tickers.")
            logging.info(f"Daily download limit reached. Skipping the rest of the tickers.")
        

    save_config(config)
    print("Configuration updated.")
    logging.info("Configuration updated.")

if __name__ == '__main__':
    main()
