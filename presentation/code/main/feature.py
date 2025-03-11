import pandas as pd
import numpy as np

class FeatureType:
    BINARY = 'binary'
    CONTINUOUS = 'continuous'

class FeatureGenerator:
    def __init__(self):
        """
        Initializes the FeatureGenerator with default parameters.
        Each key represents a feature name with sub-properties.
        For features that depend on a period, the period property can be a list.
        """
        self.params = {
            'Binary_SMA_Crossover': {
                'sma_short_period': [120, 240, 480, 960, 1920],       # list of short-term periods
                'sma_long_period': [240, 480, 960, 1920, 3840],         # list of long-term periods (should have same length as short list)
                'is_binary': True                           # binary output
            },
            'Price_Above_VWAP': {
                'vwap_period': [120, 240, 480, 960, 1920],
                'is_binary': True
            },
            'MACD_Bullish': {
                'macd_fast_period': [120, 240, 480, 960, 1920],
                'macd_slow_period': [240, 480, 960, 1920, 3840],
                'macd_signal_period': [90, 120, 240, 480, 960],
                'is_binary': True
            },
            'RSI': {
                'rsi_period': [120, 240, 480, 960, 1920],
                'is_binary': False
            },
            'RSI_Threshold': {
                'rsi_lower': [15, 20, 25, 30, 35],
                'rsi_upper': [85, 80, 75, 70, 65],
                'is_binary': True
            },
            'BB_Breakout': {
                'boll_period': [120, 240, 480, 960, 1920],
                'is_binary': True
            },
            'SMA_cont': {
                'sma_cont_period': [120, 240, 480, 960, 1920],
                'is_binary': False
            },
            'ATR': {
                'atr_period': [120, 240, 480, 960, 1920],
                'is_binary': False
            },
            'Momentum': {
                'momentum_period': [120, 240, 480, 960, 1920],
                'is_binary': False
            },
            'Average_Return': {
                'avg_return_period': [120, 240, 480, 960, 1920],
                'is_binary': False
            },
            'Rate_Close_Greater_Open': {
                'rate_period': [120, 240, 480, 960, 1920],
                'is_binary': False
            },
            'Downside_Deviation': {
                'dd_period': [120, 240, 480, 960, 1920],
                'is_binary': False
            },
            'Sortino_Ratio': {
                'sortino_period': [120, 240, 480, 960, 1920],
                'is_binary': False
            },
            'Max_Close': {
                'max_close_period': [120, 240, 480, 960, 1920],
                'is_binary': False
            },
            'Min_Close': {
                'min_close_period': [120, 240, 480, 960, 1920],
                'is_binary': False
            }
        }
        # This list will accumulate names of intermediate columns (with period suffixes) to be dropped later.
        self._intermediate_columns = []
        self._feature_names = []
        self._feature_types = {}
        
        # List of feature methods to apply.
        self.feature_functions = [
            self._binary_sma_crossover,
            self._price_above_vwap,
            self._macd_bullish,
            self._rsi,           # generates RSI columns
            self._rsi_threshold, # uses RSI columns to generate thresholds
            self._bb_breakout,
            self._sma_cont,
            self._bollinger_width,
            self._atr,
            self._momentum,
            self._avg_return,
            self._rate_close_gt_open,
            self._downside_deviation,
            self._sortino_ratio,
            self._max_close,           # generates max close columns
            self._min_close,
            self._close,
            self._open,
            self._high,
            self._low,
            self._volume
        ]
        
    def generate(self, df):
        """
        Generate all features based on the internally defined parameters.
        
        Parameters:
            df (pd.DataFrame): DataFrame with at least 'close', 'high', 'low', 'volume'.
        
        Returns:
            pd.DataFrame: DataFrame with new feature columns.
        """
        df = df.copy().sort_index(ascending=True)
        for func in self.feature_functions:
            df = func(df)
        df = self._drop_intermediate_columns(df)
        return df

    def _binary_sma_crossover(self, df):
        params = self.params['Binary_SMA_Crossover']
        # Ensure parameters are lists
        short_list = params.get('sma_short_period', [5])
        long_list = params.get('sma_long_period', [20])
        # Iterate over paired values (assumes equal length)
        for s, l in zip(short_list, long_list):
            col_short = f"SMA_short_{s}"
            col_long = f"SMA_long_{l}"
            feature_name = f"Binary_SMA_Crossover_{s}_{l}"
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.BINARY
            df[col_short] = df['close'].rolling(window=s, min_periods=s).mean()
            df[col_long] = df['close'].rolling(window=l, min_periods=l).mean()
            df[feature_name] = (df[col_short] > df[col_long]).astype(int)
            self._intermediate_columns.extend([col_short, col_long])
        return df

    def _price_above_vwap(self, df):
        params = self.params['Price_Above_VWAP']
        period_list = params.get('vwap_period', [20])
        for period in period_list:
            col_pv = f"pv_{period}"
            col_rpv = f"rolling_pv_{period}"
            col_rvol = f"rolling_volume_{period}"
            col_vwap = f"VWAP_{period}"
            feature_name = f"Price_Above_VWAP_{period}"
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.BINARY
            df[col_pv] = df['close'] * df['volume']
            df[col_rpv] = df[col_pv].rolling(window=period, min_periods=period).sum()
            df[col_rvol] = df['volume'].rolling(window=period, min_periods=period).sum()
            df[col_vwap] = df[col_rpv] / df[col_rvol]
            df[feature_name] = (df['close'] > df[col_vwap]).astype(int)
            self._intermediate_columns.extend([col_pv, col_rpv, col_rvol, col_vwap])
        return df

    def _macd_bullish(self, df):
        params = self.params['MACD_Bullish']
        fast_list = params.get('macd_fast_period', [12])
        slow_list = params.get('macd_slow_period', [26])
        signal_list = params.get('macd_signal_period', [9])
        for fast, slow, signal in zip(fast_list, slow_list, signal_list):
            col_fast = f"EMA_fast_{fast}"
            col_slow = f"EMA_slow_{slow}"
            col_macd = f"MACD_{fast}_{slow}"
            col_signal = f"MACD_Signal_{signal}"
            feature_name = f"MACD_Bullish_{fast}_{slow}_{signal}"
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.BINARY
            df[col_fast] = df['close'].ewm(span=fast, adjust=False).mean()
            df[col_slow] = df['close'].ewm(span=slow, adjust=False).mean()
            df[col_macd] = df[col_fast] - df[col_slow]
            df[col_signal] = df[col_macd].ewm(span=signal, adjust=False).mean()
            df[feature_name] = ((df[col_macd] > df[col_signal]) & 
                                (df[col_macd].shift(1) <= df[col_signal].shift(1))).astype(int)
            self._intermediate_columns.extend([col_fast, col_slow, col_macd, col_signal])
        return df

    def _rsi(self, df):
        params = self.params['RSI']
        period_list = params.get('rsi_period', [10])
        for period in period_list:
            feature_name = f"RSI_{period}"
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.CONTINUOUS
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()
            rs = avg_gain / avg_loss
            df[feature_name] = 100 - (100 / (1 + rs))
        return df

    def _rsi_threshold(self, df):
        params = self.params['RSI_Threshold']
        rsi_lower_list = params.get('rsi_lower', [30])
        rsi_upper_list = params.get('rsi_upper', [70])
        rsi_params = self.params['RSI']
        period_list = rsi_params.get('rsi_period', [10])

        for low, up in zip(rsi_lower_list, rsi_upper_list):
            for period in period_list:
                feature_name = f"RSI_Threshold_{period}_{low}_{up}"
                self._feature_names.append(feature_name)
                self._feature_types[feature_name] = FeatureType.BINARY
                delta = df['close'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=period, min_periods=period).mean()
                avg_loss = loss.rolling(window=period, min_periods=period).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                df[feature_name] = ((rsi < low) | (rsi > up)).astype(int)
        return df

    def _bb_breakout(self, df):
        params = self.params['BB_Breakout']
        period_list = params.get('boll_period', [20])
        for period in period_list:
            col_sma = f"SMA_boll_{period}"
            col_std = f"std_boll_{period}"
            col_upper = f"Upper_Band_{period}"
            col_lower = f"Lower_Band_{period}"
            feature_name = f"BB_Breakout_{period}"
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.BINARY
            df[col_sma] = df['close'].rolling(window=period, min_periods=period).mean()
            df[col_std] = df['close'].rolling(window=period, min_periods=period).std()
            df[col_upper] = df[col_sma] + 2 * df[col_std]
            df[col_lower] = df[col_sma] - 2 * df[col_std]
            df[feature_name] = ((df['close'] > df[col_upper]) | (df['close'] < df[col_lower])).astype(int)
            self._intermediate_columns.extend([col_sma, col_std, col_upper, col_lower])
        return df

    def _sma_cont(self, df):
        params = self.params['SMA_cont']
        period_list = params.get('sma_cont_period', [10])
        for period in period_list:
            feature_name = f"SMA_cont_{period}"
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.CONTINUOUS
            df[feature_name] = df['close'].rolling(window=period, min_periods=period).mean()
        return df

    def _bollinger_width(self, df):
        # Use the same periods as in BB_Breakout to compute Bollinger Width.
        params = self.params['BB_Breakout']
        period_list = params.get('boll_period', [20])
        for period in period_list:
            col_sma = f"SMA_boll_{period}"
            col_upper = f"Upper_Band_{period}"
            col_lower = f"Lower_Band_{period}"
            feature_name = f"Bollinger_Width_{period}"
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.CONTINUOUS
            df[feature_name] = (df[col_upper] - df[col_lower]) / df[col_sma]
        return df

    def _atr(self, df):
        params = self.params['ATR']
        period_list = params.get('atr_period', [10])
        df['prev_close'] = df['close'].shift(1)
        # Compute TR once for all ATR calculations.
        df['TR'] = df.apply(lambda row: max(row['high'] - row['low'],
                                              abs(row['high'] - row['prev_close']) if pd.notnull(row['prev_close']) else 0,
                                              abs(row['low'] - row['prev_close']) if pd.notnull(row['prev_close']) else 0),
                            axis=1)
        for period in period_list:
            feature_name = f"ATR_{period}"
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.CONTINUOUS
            df[feature_name] = df['TR'].rolling(window=period, min_periods=period).mean()
            self._intermediate_columns.extend(['prev_close', 'TR'])
        return df

    def _momentum(self, df):
        params = self.params['Momentum']
        period_list = params.get('momentum_period', [10])
        for period in period_list:
            feature_name = f"Momentum_{period}"
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.CONTINUOUS
            df[feature_name] = df['close'] - df['close'].shift(period)
        return df
    
    def _avg_return(self, df):
        """
        Computes the average return over a given list of periods.
        For each period n, the feature is computed over a rolling window of length n as:
        
            Average_Return_n = mean( (price[i] - price[0]) / price[0] for i = 1, ..., n-1 )
        
        This gives the average return relative to the first price in the window.
        """
        params = self.params['Average_Return']
        period_list = params.get('avg_return_period', [120])
        
        # Define a helper function to compute average return for a given window.
        def compute_avg_return(window):
            base = window[0]
            # Avoid division by zero
            if base == 0:
                return np.nan
            returns = [(w - base) / base for w in window[1:]]
            return sum(returns) / len(returns)
        
        for period in period_list:
            feature_name = f"Average_Return_{period}"
            # Apply the custom function on a rolling window of 'close' prices.
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.CONTINUOUS
            df[feature_name] = df['close'].rolling(window=period, min_periods=period).apply(compute_avg_return, raw=True)
        return df
    
    def _rate_close_gt_open(self, df):
        """
        Computes the rate (proportion) of times that the close price is greater than the open price
        over a rolling window defined by each period in the 'rate_period' list.
        """
        params = self.params['Rate_Close_Greater_Open']
        period_list = params.get('rate_period', [5])
        for period in period_list:
            feature_name = f"Rate_Close_Gt_Open_{period}"
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.CONTINUOUS
            indicator = (df['close'] > df['open']).astype(int)
            df[feature_name] = indicator.rolling(window=period, min_periods=period).mean()
        return df

    def _downside_deviation(self, df):
        """
        Computes the downside deviation of the percentage returns over a rolling window.
        For each period n, it calculates:
        
            Downside_Deviation_n = sqrt(mean( min(0, ret)^2 ))
        
        where ret is the percentage return, and only negative returns contribute.
        """
        params = self.params['Downside_Deviation']
        period_list = params.get('dd_period', [10])
        returns = df['close'].pct_change()
        for period in period_list:
            feature_name = f"Downside_Deviation_{period}"
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.CONTINUOUS
            # Use rolling apply to compute downside deviation for each window.
            df[feature_name] = returns.rolling(window=period, min_periods=period).apply(
                lambda x: np.sqrt(np.mean(np.minimum(x, 0)**2)),
                raw=True
            )
        return df
    
    def _sortino_ratio(self, df):
        """
        Computes a simplified Sortino ratio over a rolling window.
        For each period n, it is defined as:
            Sortino_Ratio_n = (rolling average return) / (rolling downside deviation)
        where rolling average return is the mean of percentage returns over the window,
        and rolling downside deviation is computed as in _downside_deviation.
        """
        params = self.params['Sortino_Ratio']
        period_list = params.get('sortino_period', [10])
        returns = df['close'].pct_change()
        for period in period_list:
            feature_name = f"Sortino_Ratio_{period}"
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.CONTINUOUS
            avg_ret = returns.rolling(window=period, min_periods=period).mean()
            downside = returns.rolling(window=period, min_periods=period).apply(
                lambda x: np.sqrt(np.mean(np.minimum(x, 0)**2)),
                raw=True
            )
            # Avoid division by zero by setting result to NaN if downside is 0.
            sortino = avg_ret / downside
            df[feature_name] = sortino
        return df

    def _max_close(self, df):
        """
        Computes the maximum close price over a rolling window.
        For each period n specified in 'Max_Close', a new feature column is generated.
        """
        params = self.params['Max_Close']
        period_list = params.get('max_close_period', [10])
        for period in period_list:
            feature_name = f"Max_Close_{period}"
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.CONTINUOUS
            df[feature_name] = df['close'].rolling(window=period, min_periods=period).max()
        return df
    
    def _min_close(self, df):
        """
        Computes the minimum close price over a rolling window.
        For each period n specified in 'Min_Close', a new feature column is generated.
        """
        params = self.params['Min_Close']
        period_list = params.get('min_close_period', [10])
        for period in period_list:
            feature_name = f"Min_Close_{period}"
            self._feature_names.append(feature_name)
            self._feature_types[feature_name] = FeatureType.CONTINUOUS
            df[feature_name] = df['close'].rolling(window=period, min_periods=period).min()
        return df

    def _close(self, df):
        self._feature_names.append('close')
        self._feature_types['close'] = FeatureType.CONTINUOUS
        return df
    
    def _open(self, df):
        self._feature_names.append('open')
        self._feature_types['open'] = FeatureType.CONTINUOUS
        return df
    
    def _high(self, df):
        self._feature_names.append('high')
        self._feature_types['high'] = FeatureType.CONTINUOUS
        return df
    
    def _low(self, df):
        self._feature_names.append('low')
        self._feature_types['low'] = FeatureType.CONTINUOUS
        return df
    
    def _volume(self, df):
        self._feature_names.append('volume')
        self._feature_types['volume'] = FeatureType.CONTINUOUS
        return df

    def _drop_intermediate_columns(self, df):
        df.drop(columns=list(set(self._intermediate_columns)), inplace=True, errors='ignore')
        return df
    
    def _generate_target_Y_fast(self, data, future_periods=4, threshold = 0.01):
        """
        Generate a binary target variable Y in a vectorized manner.

        For each observation:
        Y = 1 if (average of the close prices in the next 'future_hours' rows - current close price) * 100 / current close price > 0.01
        Y = 0 otherwise.

        Parameters:
            df (pd.DataFrame): DataFrame with a 'close' column, indexed by datetime.
            future_hours (int): Number of future rows to average (e.g., 4 for the next 4 hours).

        Returns:
            pd.DataFrame: A DataFrame with an added binary 'Y' column.
                        Rows without enough future data are assigned NaN.
        """
        df = data.copy()
        n = future_periods

        # Compute the average of the next n close prices.
        # Step 1: Shift the 'close' column by -1 so that the next row's value aligns with the current row.
        # Step 2: Compute a rolling average of the next n values.
        # Step 3: Shift the result upward by (n - 1) rows to align it with the current row.
        future_avg = df.sort_index(ascending=True)['close'].shift(-1).rolling(window=n, min_periods=n).mean().shift(-(n - 1))

        # Compute the percentage change between the future average and current close.
        percentage_change = (future_avg - df['close']) * 100 / df['close']

        # Create the binary target: 1 if percentage_change > 0.01, else 0.
        # If percentage_change is NaN (due to insufficient future data), Y will be set to NaN.

        df['Y'] = (percentage_change > threshold).astype(int)
        df.loc[percentage_change.isna(), 'Y'] = 0

        return df

# --- Example Usage ---
if __name__ == "__main__":
    # np.random.seed(42)
    # dates = pd.date_range('2023-01-01', periods=100, freq='H')
    # data = {
    #     'close': 100 * np.cumprod(1 + np.random.normal(0, 0.001, 100)),
    #     'high': 100 * np.cumprod(1 + np.random.normal(0, 0.001, 100)) * 1.01,
    #     'low': 100 * np.cumprod(1 + np.random.normal(0, 0.001, 100)) * 0.99,
    #     'volume': np.random.randint(100, 200, 100)
    # }
    # df_sample = pd.DataFrame(data, index=dates)
    
    from setup import DataLoader
    data_loader = DataLoader(is_google_colab=False)
    file_dict = data_loader.get_data_filepath_dict()
    # file_path = file_dict['META']

    # Create an instance of FeatureGenerator and generate features.
    data = data_loader.load_data(file_path=file_dict['META'], date_column_name='timestamp')

    feature_generator = FeatureGenerator()
    df_features = feature_generator.generate(data)
    print(df_features.head(1000).tail(5))
    print(feature_generator._feature_names)
    # data_features_filepath = os.path.join(project_dir, 'data_features.csv')
    # df_features.to_csv(data_features_filepath)
