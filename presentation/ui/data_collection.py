import csv
import requests
import json
import pandas as pd
# import ace_tools as tools

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

API_KEY = config["api_key"]
CSV_URL = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={API_KEY}'

# Fetch the stock listing data
with requests.Session() as s:
    download = s.get(CSV_URL)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    stock_list = list(cr)

# Convert stock data to DataFrame
df_stocks = pd.DataFrame(stock_list[1:], columns=stock_list[0])

# Fetch sector and industry info for the first 5 stocks to avoid API rate limits
overview_data = []
for symbol in df_stocks['symbol'].head(5):
    OVERVIEW_URL = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={API_KEY}'
    print(OVERVIEW_URL)
    response = requests.get(OVERVIEW_URL)
    if response.status_code == 200:
        data = response.json()
        overview_data.append({
            'Symbol': symbol,
            'Sector': data.get('Sector', 'N/A'),
            'Industry': data.get('Industry', 'N/A'),
            'MarketCap': data.get('MarketCapitalization', 'N/A')
        })

# Convert overview data to DataFrame
df_overview = pd.DataFrame(overview_data)

# Merge stock listing with overview data
df_combined = pd.merge(df_stocks, df_overview, left_on='symbol', right_on='Symbol', how='left')
print(df_combined.head(5))