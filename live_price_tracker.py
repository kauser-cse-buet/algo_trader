import requests
import streamlit as st
import time
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import matplotlib.dates as mdates
import pandas as pd

# Function to fetch live or historical price
def get_price(symbol, is_crypto=False):
    api_key = 'cujh6j1r01qm7p9o4ql0cujh6j1r01qm7p9o4qlg'  # Replace with your actual API key from Finnhub
    if is_crypto:
        api_url = f'https://finnhub.io/api/v1/crypto/quote?symbol={symbol}&token={api_key}'
    else:
        api_url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}'

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        return data.get('c')  # 'c' stands for current price
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to fetch historical data for stocks using Yahoo Finance
def get_historical_data(symbol, start_date, end_date, is_crypto=False):
    if is_crypto:
        api_key = 'cujh6j1r01qm7p9o4ql0cujh6j1r01qm7p9o4qlg'
        resolution = 'D'
        start_unix = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_unix = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        api_url = f'https://finnhub.io/api/v1/crypto/candle?symbol={symbol}&resolution={resolution}&from={start_unix}&to={end_unix}&token={api_key}'

        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            if data.get('s') == 'ok':
                return data['t'], data['c']  # Timestamps and closing prices
            else:
                return [], []
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching historical data: {e}")
            return [], []
    else:
        # Fetch historical stock data using Yahoo Finance
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if not data.empty:
                timestamps = data.index.astype('int64') // 10**9  # Convert to UNIX timestamp
                prices = data['Close'].tolist()
                return timestamps.tolist(), prices, data
            else:
                return [], [], pd.DataFrame()
        except Exception as e:
            st.error(f"Error fetching historical data: {e}")
            return [], [], pd.DataFrame()

# Backtesting Strategy: Simple Moving Average Crossover
def backtest_strategy(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    data['Signal'] = 0
    data.loc[data['SMA_20'] > data['SMA_50'], 'Signal'] = 1
    data.loc[data['SMA_20'] <= data['SMA_50'], 'Signal'] = -1

    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']

    cumulative_returns = (1 + data['Strategy_Returns']).cumprod()
    return data, cumulative_returns

# Streamlit UI
st.title('📈 Price Tracker & Backtester')

asset_type = st.selectbox('Select Asset Type:', ('Stock', 'Cryptocurrency'))
symbol = st.text_input('Enter Symbol (e.g., AAPL for Stock, BINANCE:BTCUSDT for Crypto):', 'AAPL')

view_mode = st.radio('Select View Mode:', ('Live Price', 'Historical Price', 'Backtest Strategy'))

if view_mode == 'Live Price':
    interval = st.slider('Update Interval (seconds):', min_value=1, max_value=60, value=5)
    start_tracking = st.button('Start Tracking', key='start_button')

    if start_tracking:
        placeholder = st.empty()
        stop_button_placeholder = st.empty()
        prices = []
        timestamps = []

        stop_tracking = False

        while not stop_tracking:
            is_crypto = asset_type == 'Cryptocurrency'
            price = get_price(symbol, is_crypto)
            if price:
                prices.append(price)
                timestamps.append(time.strftime('%H:%M:%S'))

                # Maintain a rolling window of the last 8 data points
                if len(prices) > 8:
                    prices = prices[-8:]
                    timestamps = timestamps[-8:]

                # Plotting the price
                plt.figure(figsize=(10, 5))
                plt.plot(timestamps, prices, marker='o')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.title(f'Live Price of {symbol.upper()}')
                plt.grid(True)

                placeholder.pyplot(plt)
                plt.close()

            else:
                placeholder.markdown(f"### ⚠️ Failed to retrieve price for **{symbol.upper()}**")

            time.sleep(interval)

            if stop_button_placeholder.button('Stop Tracking', key=f'stop_button_{time.time()}'):
                stop_tracking = True

elif view_mode == 'Historical Price':
    start_date = st.date_input('Start Date', value=datetime(2023, 1, 1))
    end_date = st.date_input('End Date', value=datetime.now())

    if st.button('Show Historical Data'):
        is_crypto = asset_type == 'Cryptocurrency'
        timestamps, prices, data = get_historical_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), is_crypto)

        if timestamps and prices:
            formatted_dates = [datetime.fromtimestamp(ts) for ts in timestamps]

            plt.figure(figsize=(14, 6))
            plt.plot(formatted_dates, prices, marker='o', linestyle='-', linewidth=2, markersize=4)
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Historical Price of {symbol.upper()}')
            plt.grid(True, linestyle='--', alpha=0.6)

            # Improve x-axis formatting
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45, ha='right')

            st.pyplot(plt)
            plt.close()

        else:
            st.warning(f"No historical data available for **{symbol.upper()}** in the selected range.")

elif view_mode == 'Backtest Strategy':
    start_date = st.date_input('Start Date', value=datetime(2023, 1, 1))
    end_date = st.date_input('End Date', value=datetime.now())

    if st.button('Run Backtest'):
        is_crypto = asset_type == 'Cryptocurrency'
        _, _, data = get_historical_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), is_crypto)

        if not data.empty:
            data, cumulative_returns = backtest_strategy(data)

            plt.figure(figsize=(14, 6))
            plt.plot(data.index, cumulative_returns, label='Strategy Cumulative Returns', linewidth=2)
            plt.plot(data.index, (1 + data['Returns']).cumprod(), label='Buy and Hold Returns', linestyle='--')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.title(f'Backtest Results for {symbol.upper()}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            st.pyplot(plt)
            plt.close()

        else:
            st.warning(f"No historical data available for **{symbol.upper()}** in the selected range.")
