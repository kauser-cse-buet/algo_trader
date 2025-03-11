import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class StrategyManager:
    def simulate_strategy(self, df, threshold, holding_period, take_profit, stop_loss, signal_col_name='predicted_signal'):
        """
        Simulates trades based on a threshold for the predicted signal. A trade is entered
        when 'predicted_signal' >= threshold. The trade is held for up to 'holding_period'
        periods, exiting if take profit or stop loss is hit.

        Returns:
            trade_df (pd.DataFrame): Details of each trade
            metrics (dict): { 'num_trades', 'win_rate', 'total_return', 'sharpe_ratio' }
        """
        trades = []
        i = 0
        n = len(df)
        while i < n:
            if df.iloc[i][signal_col_name]:
                entry_price = df.iloc[i]['close']
                exit_price = None
                exit_reason = None
                entry_index = i

                # Simulate holding
                for j in range(i+1, min(i + 1 + holding_period, n)):
                    current_price = df.iloc[j]['close']
                    ret = (current_price - entry_price) / entry_price
                    # Check take profit
                    if ret >= take_profit:
                        exit_price = current_price
                        exit_reason = 'tp'
                        i = j + 1
                        break
                    # Check stop loss
                    if ret <= stop_loss:
                        exit_price = current_price
                        exit_reason = 'sl'
                        i = j + 1
                        break
                # If neither is triggered, exit at the end of holding period
                if exit_price is None:
                    exit_index = min(i + holding_period, n - 1)
                    exit_price = df.iloc[exit_index]['close']
                    exit_reason = 'hold'
                    i = exit_index + 1

                trade_return = (exit_price / entry_price) - 1
                trades.append({
                    'entry_index': entry_index,
                    'exit_index': i-1,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'threshold': threshold,
                    'holding_period': holding_period,
                    'take_profit': take_profit,
                    'stop_loss': stop_loss,
                    'exit_reason': exit_reason
                })
            else:
                i += 1

        if len(trades) == 0:
            return pd.DataFrame(), {
                'num_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'sharpe_ratio': np.nan
            }

        trade_df = pd.DataFrame(trades)
        # Calculate performance metrics
        num_trades = len(trade_df)
        win_rate = (trade_df['return'] > 0).mean()

        total_return = (1 + trade_df['return']).prod() - 1
        # total_return = [i['return'] * i['entry_price'] for i in trade_df].sum()
        # total_return = (trade_df['entry_price'] * trade_df['return']).sum()
        # total_return = sum([row['entry_price'] * row['return'] for index, row in trade_df.iterrows()])

        returns_std = trade_df['return'].std()
        sharpe_ratio = trade_df['return'].mean() / returns_std if returns_std != 0 else np.nan

        metrics = {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio
        }
        return trade_df, metrics

    def plot_tp_sl_horizontal(self, df, trades_df):
        """
        Plots the price series with horizontal lines for each trade's take profit and stop loss levels.
        
        Parameters:
            df (pd.DataFrame): DataFrame with a DateTime index and a 'close' column.
            trades_df (pd.DataFrame): DataFrame with trade details including:
                - 'entry_index'
                - 'exit_index'
                - 'entry_price'
                - 'take_profit'
                - 'stop_loss'
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the price series
        ax.plot(df.index, df['close'], label='Close Price', color='black')
        
        # Loop over each trade to plot TP and SL horizontal lines
        for idx, trade in trades_df.iterrows():
            # Get the entry and exit times using the indices
            entry_idx = int(trade['entry_index'])
            exit_idx = int(trade['exit_index'])
            entry_time = df.index[entry_idx]
            exit_time = df.index[exit_idx]
            entry_price = trade['entry_price']
            
            # Calculate TP and SL levels
            tp_level = entry_price * (1 + trade['take_profit'])
            sl_level = entry_price * (1 + trade['stop_loss'])
            
            # Plot horizontal lines for TP and SL across the trade holding period
            ax.hlines(tp_level, xmin=entry_time, xmax=exit_time, colors='blue', linestyles='dotted',
                    label='Take Profit' if idx == 0 else "")
            ax.hlines(sl_level, xmin=entry_time, xmax=exit_time, colors='orange', linestyles='dotted',
                    label='Stop Loss' if idx == 0 else "")
            
            # Optionally, mark the trade's holding period with a vertical span for clarity
            ax.axvspan(entry_time, exit_time, color='gray', alpha=0.1)
        
        ax.set_title("Price Series with Take Profit and Stop Loss Levels")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        plt.show()


    def grid_search_trade_setups(
        self,
        df,
        holding_periods,
        take_profits,
        stop_losses,
        thresholds,
        verbose = False
    ):
        """
        Tests all combinations of holding periods, take profits, stop losses, and thresholds.
        Returns a DataFrame summarizing performance for each combination.

        Parameters:
            df (pd.DataFrame): Must contain columns 'close' and 'predicted_signal'.
            holding_periods (list): List of possible holding periods.
            take_profits (list): List of take profit levels (e.g. [0.005, 0.01]).
            stop_losses (list): List of stop loss levels (e.g. [-0.005, -0.01]).
            thresholds (list): List of thresholds for predicted_signal.

        Returns:
            results_df (pd.DataFrame): A DataFrame with columns:
                - holding_period
                - take_profit
                - stop_loss
                - threshold
                - num_trades
                - win_rate
                - total_return
                - sharpe_ratio
        """
        results = []
        for hp in holding_periods:
            for tp in take_profits:
                for sl in stop_losses:
                    for th in thresholds:
                        df['predicted_signal'] = (df['predicted_signal_prob'] >= th).astype(int)
                        trade_df, metrics = self.simulate_strategy(df, th, hp, tp, sl, signal_col_name='predicted_signal')
                        trade_df_actual, metrics_actual = self.simulate_strategy(df, th, hp, tp, sl, signal_col_name='actual_signal')
                        result_row = {
                            'holding_period': hp,
                            'take_profit': tp,
                            'stop_loss': sl,
                            'threshold': th,
                            'num_trades': metrics['num_trades'],
                            'win_rate': metrics['win_rate'],
                            'total_return': metrics['total_return'],
                            'sharpe_ratio': metrics['sharpe_ratio'],
                            'num_trades_actual': metrics_actual['num_trades'],
                            'win_rate_actual': metrics_actual['win_rate'],
                            'total_return_actual': metrics_actual['total_return'],
                            'sharpe_ratio_actual': metrics_actual['sharpe_ratio']
                        }
                        if verbose:
                            print(result_row)
                        results.append(result_row)
                        # plot_tp_sl_horizontal(df, trade_df[:100])
                        # break

        results_df = pd.DataFrame(results)
        return results_df

    def get_unscaled_close_price(self, scaled_df, unscaled_df, timestamp_column='timestamp', close_column='close'):
        """
        Retrieves the close price from the unscaled DataFrame for a given timestamp in the scaled DataFrame.

        Args:
            scaled_df (pd.DataFrame): DataFrame containing scaled data and a timestamp column.
            unscaled_df (pd.DataFrame): DataFrame containing unscaled data and a timestamp column.
            timestamp_column (str): Name of the timestamp column (default: 'timestamp').
            close_column (str): Name of the close price column (default: 'close').

        Returns:
            float or None: The close price from the unscaled DataFrame for the corresponding timestamp,
                        or None if the timestamp is not found.
        """

        # Ensure the timestamp columns are of datetime objects
        # scaled_df[timestamp_column] = pd.to_datetime(scaled_df.index)
        # unscaled_df[timestamp_column] = pd.to_datetime(unscaled_df.index)

        # Merge the two DataFrames based on the timestamp column

        merged_df = pd.merge(scaled_df, unscaled_df, left_index=True, right_index=True, how='left', suffixes=('_scaled', '_unscaled'))

        print(merged_df.head(5))
        # Extract the close price from the unscaled DataFrame
        close_prices = merged_df[[close_column + '_unscaled']]
        close_prices.columns = ['close']

        return close_prices


from setup import DataLoader
from model import MLModelManager
import os
# --- Example Usage ---
if __name__ == "__main__":
        # Plot model performance and mark the best configuration
    dataLoader = DataLoader(is_google_colab=False)
    data_dir = os.path.join(dataLoader.intermediate_data_dir, 'META')
    print(data_dir)
    mlModelManager = MLModelManager()
    grid_search_results_df = pd.read_csv(os.path.join(data_dir, 'grid_search_results.csv'))
    best_model_params = mlModelManager.get_best_model_params_and_plot(grid_search_results_df)
    print(f"Best Model: {best_model_params}")
    x_train = pd.read_csv(os.path.join(data_dir, 'x_train_sf.csv'), index_col='timestamp') 
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train_sf.csv'), index_col='timestamp')
    print(x_train.head(5))
    print(y_train.head(5))
    # Train the model with the best parameters
    best_model = mlModelManager.train_model_with_best_params(x_train, y_train, best_model_params)

    # Output the trained model details (or use best_model for predictions, etc.)
    print("Trained XGBoost Model with Best Parameters:")
    print(best_model)


    strategyManager = StrategyManager()


    # Define parameter grids
    holding_periods = [60, 120, 240]
    take_profits = [0.005, 0.01]
    stop_losses = [-0.005, -0.01]
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    

    x_test1_sf = pd.read_csv(os.path.join(data_dir, 'x_test1_sf.csv'), index_col='timestamp')
    x_test2_sf = pd.read_csv(os.path.join(data_dir, 'x_test2_sf.csv'), index_col='timestamp')
    test1_data = pd.read_csv(os.path.join(data_dir, 'test1_data.csv'), index_col='timestamp')
    test2_data = pd.read_csv(os.path.join(data_dir, 'test2_data.csv'), index_col='timestamp')

    x = x_test2_sf
    test_data = test2_data

    data_without_scaling = pd.read_csv(os.path.join(data_dir, 'data_features_Y.csv'), index_col='timestamp')
    test_data = strategyManager.get_unscaled_close_price(scaled_df=test_data, unscaled_df=data_without_scaling)
    
    print(test_data.head(5))


    predicted_prob = best_model.predict_proba(x)
    predicted_signal = [x for x in predicted_prob[:, 1]]
    print(predicted_prob)

    df = pd.DataFrame({
        'close': x['close'],
        'predicted_signal': predicted_signal
    }, index=x.index)

    print(df.head(5))

    # # Perform the grid search
    results_df = strategyManager.grid_search_trade_setups(
        df,
        holding_periods,
        take_profits,
        stop_losses,
        thresholds
    )

    # # Print or save results
    print("Grid Search Results:")
    print(results_df.head(15))  # Show first 15 rows

    # # Find the best combination by Sharpe ratio
    best_row = results_df.dropna(subset=['sharpe_ratio']).sort_values('sharpe_ratio', ascending=False).head(1)
    print("\nBest Combination by Sharpe Ratio:")
    print(best_row)
