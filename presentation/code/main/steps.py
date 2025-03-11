from pdb import run
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import pickle
from strategy import StrategyManager
import shap
import random
from setup import DataLoader
from exclude import ObservationExclusion
from feature import FeatureGenerator, FeatureType
from model import MLModelManager
from xgboost import XGBClassifier

class Indicator:
    Open = "open"
    Close = "close"
    High = "high"
    Low = "low"
    Volume = "volume"

def plot_hourly_volume_distribution(df, percentiles, indicator=Indicator.Volume):
    """
    Plots the hourly distribution of volume for specified percentiles.

    Args:
        data: DataFrame with 'timestamp' and 'volume' columns.
        percentiles: A list of percentiles to plot.
    """
    data = df.copy()
    data['timestamp'] = df.index
    # Convert timestamp to datetime objects if they aren't already
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])

    data['hour'] = data['timestamp'].dt.hour
    print(data.head(5))

    # Calculate hourly volume percentiles and reshape for plotting
    hourly_volume = data.groupby('hour')[indicator].agg(
        lambda x: [np.percentile(x, p) for p in percentiles]
    )
    hourly_volume = hourly_volume.apply(pd.Series)  # Convert to DataFrame
    hourly_volume.columns = [f'{p}th Percentile' for p in percentiles]  # Rename columns
    print(hourly_volume.head(5))

    plt.figure(figsize=(10, 6))

    for p in percentiles:
        plt.plot(hourly_volume.index, hourly_volume[f'{p}th Percentile'], label=f'{p}th Percentile')

    plt.xlabel('Hour of Day')
    plt.ylabel(indicator)
    plt.title(f'Hourly {indicator} Distribution by Percentile')
    plt.xticks(range(24))  # Ensure all hours are shown on x-axis
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage (assuming 'data' DataFrame is available)
# percentiles_to_plot = [10, 25, 50, 75]  # Example: plot 25th, 50th, and 75th percentiles

def plot_Y_distribution_with_frequency(df):
    """Plots the distribution of the 'Y' variable with frequency counts."""

    plt.figure(figsize=(10, 6))

    # Calculate frequency counts for each bin
    counts, bins = np.histogram(df['Y'].dropna(), bins=50)  # Adjust bins as needed

    # Plot the histogram with frequency counts
    plt.hist(df['Y'].dropna(), bins=bins, edgecolor='black')

    # Print frequency table for each bin
    # print("Bin\tFrequency")
    # for i in range(len(counts)):
    #   print(f'{bins[i]:.2f} - {bins[i+1]:.2f}\t{counts[i]}')

    plt.xlabel('Y (Percentage Increase)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Y Variable')
    plt.grid(True)
    plt.show()

def scale_features(df, skip_columns=[], method="standard"):
    """
    Scale numeric features in the DataFrame, avoiding scaling for columns specified in skip_columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame with numeric columns.
        skip_columns (list): List of column names to exclude from scaling.
        method (str): Scaling method to use. Options:
                      - "standard" for StandardScaler (default)
                      - "minmax" for MinMaxScaler.

    Returns:
        df_scaled (pd.DataFrame): DataFrame with scaled features (columns not in skip_columns).
        scaler (object): The scaler fitted on the selected columns.
    """
    # Identify numeric columns that are not in skip_columns
    columns_to_scale = [col for col in df.select_dtypes(include='number').columns if col not in skip_columns]

    df_scaled = df.copy()

    if method.lower() == "standard":
        scaler = StandardScaler()
    elif method.lower() == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported scaling method. Choose 'standard' or 'minmax'.")

    # Fit and transform the selected columns
    df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

    return df_scaled, scaler

def run_shap_analysis(df, feature_cols, target_col, test_size=0.2, random_state=42, dependence_feature=None):
    """
    Trains an XGBoost classifier using the provided features and target, computes SHAP values,
    and plots a SHAP summary plot. Optionally, it also plots a SHAP dependence plot for a given feature.

    Parameters:
        df (pd.DataFrame): The DataFrame containing features and target.
        feature_cols (list): List of column names representing the independent variables.
        target_col (str): The name of the target variable column.
        test_size (float): Proportion of data to use as test set (default 0.2).
        random_state (int): Seed for train/test split.
        dependence_feature (str, optional): A feature name to create a SHAP dependence plot.

    Returns:
        model (xgb.XGBClassifier): The trained model.
        shap_values: SHAP values computed on the test set.
    """
    # Split the data into training and test sets
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train an XGBoost classifier
    model = xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Create SHAP explainer and compute SHAP values for the test set
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Plot the SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot")
    plt.show()

    # Optionally plot SHAP dependence plot for a specific feature if provided
    if dependence_feature is not None and dependence_feature in feature_cols:
        plt.figure()
        shap.dependence_plot(dependence_feature, shap_values, X_test, show=False)
        plt.title(f"SHAP Dependence Plot for {dependence_feature}")
        plt.show()

    return model, shap_values

# def split_test_train_data(data):
#     training_data, temp_data = train_test_split(data, test_size = 0.3, random_state = 1)
#     test1_data, test2_data = train_test_split(temp_data, test_size = 0.5, random_state = 1)
#     return training_data, test1_data, test2_data

def time_train_test_split_3way(df, train_frac=0.70, test1_frac=0.15, test2_frac=0.15):
    """
    Splits a DataFrame into three parts based on time order:
      - Training set: first train_frac (e.g., 70%) of the data.
      - Test1 set: next test1_frac (e.g., 15%) of the data.
      - Test2 set: remaining test2_frac (e.g., 15%) of the data.

    Parameters:
        df (pd.DataFrame): The DataFrame to split, assumed to be sorted chronologically.
        train_frac (float): Fraction of data to use for training.
        test1_frac (float): Fraction of data to use for the first test set.
        test2_frac (float): Fraction of data to use for the second test set.

    Returns:
        train_df (pd.DataFrame): The training set.
        test1_df (pd.DataFrame): The first test set.
        test2_df (pd.DataFrame): The second test set.
    """
    n = len(df)
    train_end = int(n * train_frac)
    test1_end = int(n * (train_frac + test1_frac))

    train_df = df.iloc[:train_end]
    test1_df = df.iloc[train_end:test1_end]
    test2_df = df.iloc[test1_end:]

    return train_df, test1_df, test2_df


def get_feature_improtance(df, feature_cols, target_col, feature_importance_threshold = 0.005):
    df[feature_cols] = df[feature_cols].astype(float)
    df[target_col] = df[target_col].astype(float)
    x_train = df[feature_cols]
    y_train = df[target_col]
    default_model = xgb.XGBClassifier()
    default_model.fit(x_train, y_train)
    importance = default_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': importance})
    params = {
        'objective': 'binary:logistic',
        'learning_rate': 0.5,
        'max_depth': 4,
        'subsample': 0.5,  # 50% of observations
        'colsample_bytree': 0.5,  # 50% of features
        'scale_pos_weight': 5,  # Weight for default observations
        'eval_metric': 'logloss'
    }
    modified_model = xgb.XGBClassifier(**params)
    modified_model.fit(x_train,y_train)
    mod_importance = modified_model.feature_importances_
    mod_importance_df = pd.DataFrame({'Feature':x_train.columns, 'Importance': mod_importance})
    return importance_df, mod_importance_df

def plot_feature_importance_comparison(
        first_model_features_df, 
        second_model_features_df, 
        feature_col, 
        feature_importance_col, 
        model1_name, 
        model2_name, 
        subset_count, 
        threshold = 0.005,
        output_file_path = 'feature_importance_comparison.png'
    ):
    """
    Plots the feature importance comparison for two models.

    Parameters:
    - df_features: DataFrame containing feature names and importance values for both models.
    - importance_col1: Column name for the first model's feature importance.
    - importance_col2: Column name for the second model's feature importance.
    - model1_name: Label for the first model (default: "Default Model").
    - model2_name: Label for the second model (default: "Modified Model").
    """
    # Sort the features based on the importance of the first model for better visualization
    model1_sorted_features = first_model_features_df.sort_values(by=feature_importance_col, ascending=False)
    model2_sorted_features = second_model_features_df.sort_values(by=feature_importance_col, ascending=False)

    model1_sorted_features = model1_sorted_features.head(subset_count)
    model2_sorted_features = model2_sorted_features.head(subset_count)
    
    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot bars for the first model's importance
    plt.barh(model1_sorted_features[feature_col], model1_sorted_features[feature_importance_col], color='skyblue', label=model1_name)

    # Overlay bars for the second model's importance
    plt.barh(model2_sorted_features[feature_col], model2_sorted_features[feature_importance_col], color='orange', alpha=0.7, label=model2_name)

    # Add a vertical line at Importance value = 0.005

    plt.axvline(x=threshold, color='red', linestyle='--')
    plt.text(threshold, len(model1_sorted_features) - 1, str(threshold), color='red', ha='left', va='top')
    # Add labels, title, and legend
    plt.xlabel('Importance')
    plt.title(f'Feature Importance: {model1_name} vs {model2_name}')
    plt.legend()

    # Adjust layout for better display
    plt.tight_layout()
    plt.savefig(output_file_path)
    
    # Show the plot
    # plt.show()

def param_tuning_using_grid_search(x_train, y_train, x_test1, y_test1, x_test2, y_test2):
    param_grid = {
        'n_estimators': [50, 100, 300],            # Number of trees
        'learning_rate': [0.01, 0.1],              # Learning rates
        'subsample': [0.5, 0.8],                   # Percentage of observations per tree
        'colsample_bytree': [0.5, 1.0],            # Percentage of features per tree
        'scale_pos_weight': [1, 5, 10]             # Weight of default observations
    }
    xgb_model = XGBClassifier(objective = 'binary:logistic', eval_metric = 'auc')
    grid_search = GridSearchCV(estimator = xgb_model, param_grid = param_grid, scoring = 'roc_auc', cv = 3, verbose = 1, n_jobs = -1)
    grid_search.fit(x_train, y_train)
    results_df = pd.DataFrame(columns = ['Trees', 'Learning Rate', 'Subsample', '% Features', 'Weight of Default', 'AUC Train', 'AUC Test 1', 'AUC Test 2'])

    trained_models = []
    train_pred_probs = []
    test1_pred_probs = []
    test2_pred_probs = []

    for params, mean_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
        best_model = XGBClassifier(**params)
        best_model.fit(x_train,y_train)
        trained_models.append(best_model)

        train_pred = best_model.predict_proba(x_train)[:, 1]
        test1_pred = best_model.predict_proba(x_test1)[:, 1]
        test2_pred = best_model.predict_proba(x_test2)[:, 1]

        auc_train = roc_auc_score(y_train, train_pred)
        auc_test1 = roc_auc_score(y_test1, test1_pred)
        auc_test2 = roc_auc_score(y_test2, test2_pred)

        train_pred_probs.append(train_pred)
        test1_pred_probs.append(test1_pred)
        test2_pred_probs.append(test2_pred)

        new_row = pd.DataFrame({
            'Trees': [params['n_estimators']],
            'Learning Rate': [params['learning_rate']],
            'Subsample': [params['subsample']],
            '% Features': [params['colsample_bytree']],
            'Weight of Default': [params['scale_pos_weight']],
            'AUC Train': [auc_train],
            'AUC Test 1': [auc_test1],
            'AUC Test 2': [auc_test2]
        })

        results_df = pd.concat([results_df, new_row], ignore_index = True)

    return results_df



class FILENAME:
    DATA_AFTER_EXCLUSION = 'data_after_exclusion.csv'
    DATA_FEATURES = 'data_features.csv'
    DATA_FEATURES_Y = 'data_features_Y.csv'
    DATA_FEATURES_Y_SCALED = 'data_features_Y_scaled.csv'
    DATA_FEATURES_Y_SCALED_TRAIN = 'data_features_Y_scaled_train.csv'
    DATA_FEATURES_Y_SCALED_TEST1 = 'data_features_Y_scaled_test1.csv'
    DATA_FEATURES_Y_SCALED_TEST2 = 'data_features_Y_scaled_test2.csv'
    DATA_TRAIN_SF = 'data_train_sf.csv'
    DATA_TEST1_SF = 'data_test1_sf.csv'
    DATA_TEST2_SF = 'data_test2_sf.csv'
    FEATURE_INFO = 'feature_info.csv'
    DEFAULT_FEATURE_IMPORTANCE = 'default_feature_importance.csv'
    MODIFIED_FEATURE_IMPORTANCE = 'modified_feature_importance.csv'
    SIGNIFICANT_FEATURES = 'shared_features.csv'
    BEST_MODEL_PARAMS = 'best_model_params.csv'
    BEST_MODEL_PARAMS_PLOT = 'best_model_params_plot.png'
    BEST_XGBOOST_MODEL = 'best_xgboost_model.pkl'
    GRID_SEARCH_RESULTS = 'grid_search_results.csv'
    STRATEGY_RESULT_TEST_1 = 'strategy_result_test_1.csv'
    STRATEGY_RESULT_TEST_2 = 'strategy_result_test_2.csv'
    BEST_STRATEGY_RESULT_TEST_1 = 'best_strategy_result_test_1.csv'
    BEST_STRATEGY_RESULT_TEST_2 = 'best_strategy_result_test_2.csv'
    SHAP_BEESWARM_TEST1 = 'SHAP_BEESWARM_TEST1.png'
    SHAP_BEESWARM_TEST2 = 'SHAP_BEESWARM_TEST2.png'
    SHAP_WATERFALL_TEST1 = 'SHAP_WATERFALL_TEST1.png'
    SHAP_WATERFALL_TEST2 = 'SHAP_WATERFALL_TEST2.png'
    

class RUN_SETUP:
    def __init__(self, is_google_colab = False, file_suffix = 'META'):
        self.line_border_len = 100
        self.date_column_name = 'timestamp'
        self.target_column_name = 'Y'
        self.file_suffix = file_suffix
        self.is_google_colab = is_google_colab
        
        data_loader = DataLoader(is_google_colab=is_google_colab)
        self.file_dict = data_loader.get_data_filepath_dict()
        self.file_path=self.file_dict[self.file_suffix]
        
        self.file_suffix_dir = os.path.join(data_loader.intermediate_data_dir, self.file_suffix)
        if not os.path.exists(self.file_suffix_dir):
            os.makedirs(self.file_suffix_dir)
        
        self.params = {
            '_load_data': { 
                'run': True,
                'description': "Load data",
                'save': False,
                'load': False
            },
            '_exclude_observation': {
                'run': True,
                'description': "Exclude observation",
                'save': True,
                'load': False,
                'output_file_name': FILENAME.DATA_AFTER_EXCLUSION
            },
            '_generate_features': {
                'run': True,
                'description': "Generate features",
                'save': True,
                'load': False,
                'output_file_name': FILENAME.DATA_FEATURES,
                'feature_info_file_name': FILENAME.FEATURE_INFO
            },
            '_generate_Y': {
                'run': True,
                'description': "Generate Y value",
                'save': True,
                'load': False,
                'output_file_name': FILENAME.DATA_FEATURES_Y,
                'future_periods': 120, 
                'threshold': 0.01
            },
            '_scale_features': {
                'run': True,
                'description': "Scale features",
                'save': True,
                'load': False,
                'output_file_name': FILENAME.DATA_FEATURES_Y_SCALED
            },
            '_split_data': {
                'run': True,
                'description': "Split data into training, test1, test2",
                'save': False,
                'load': False,
                'load_file_name': FILENAME.DATA_FEATURES_Y_SCALED
            },
            '_get_feature_importance': {
                'run': True,
                'description': "Get feature importance",
                'save': True,
                'load': False,
                'load_file_name': FILENAME.DATA_FEATURES_Y_SCALED_TRAIN,
                'output_file_name': FILENAME.DATA_TRAIN_SF,
                'feature_importance_comparison_plot': 'feature_importance_comparison.png'
            },
            '_get_significant_features':{
                'run': True,
                'description': "Get Significant features",
                'save': False,
                'load': False,
                'feature_importance_threshold': 0.0145,
                'feature_importance_comparison_plot': 'feature_importance_comparison.png'
            },
            '_param_tuning_using_grid_search': {
                'run': True,
                'description': "Parameter tuning using grid search",
                'save': False,
                'load': False,
                'load_file_name': FILENAME.DATA_TRAIN_SF
            },
            '_get_best_model_params': {
                'run': True,
                'description': "Get best model parameters",
                'save': False,
                'load': False,
                'load_file_name': FILENAME.DATA_TRAIN_SF
            },
            '_run_shap_analysis': {
                'run': True,
                'description': "Run SHAP analysis",
                'save': False,
                'load': False
            },
            '_run_strategy': {
                'run': True,
                'description': "Run strategy",
                'save': False,
                'load': False,
                'holding_periods': [60, 120, 240],
                'take_profits': [0.005, 0.01],
                'stop_losses':[-0.005, -0.01],
                'thresholds':[0.5, 0.6, 0.7, 0.8, 0.9],
                'amount': [0.1, 0.2, 0.3]
            }
        }

        self._functions = [
            # self._load_data,
            # self._exclude_observation,
            # self._generate_features,
            # self._generate_Y,
            # self._scale_features,
            # self._split_data,
            # self._get_feature_importance,
            # self._get_significant_features,
            # self._param_tuning_using_grid_search,
            # self._get_best_model_params,
            # self._run_strategy,
            self._run_shap_analysis
        ]
    
    def _get_full_file_path(self, file_name):
        return os.path.join(self.file_suffix_dir, file_name)
    
    def _shap_beeswarm_analysis(self, model, x_test, filename=FILENAME.SHAP_BEESWARM_TEST1):
        explainer = shap.Explainer(model, x_test)
        shap_values = explainer(x_test)
        plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(shap_values,  max_display=20, show=False)
        plt.savefig(self._get_full_file_path(filename), dpi=300, bbox_inches='tight')
        plt.close()
        return shap_values

    def _shap_waterfall_analysis(self, shap_values, filename, observation_count = 5):
        random_numbers = random.sample(range(len(shap_values)), observation_count)
        base_filename, extension = os.path.splitext(filename)
        # Create new filenames by appending each random number to the base filename
        for num in random_numbers:
            new_filename = f"{base_filename}_{num}{extension}"
            plt.figure(figsize=(8, 6))
            shap.plots.waterfall(shap_values[num], max_display=20, show=False)
            plt.savefig(self._get_full_file_path(new_filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f'File saved to: {new_filename}')

    
    def _run_shap_analysis(self, data, param):
        best_model = self._load_model(self._get_full_file_path(FILENAME.BEST_XGBOOST_MODEL))
        test1_data = self._load(file_name=FILENAME.DATA_FEATURES_Y_SCALED_TEST1, index_col=self.date_column_name)
        test2_data = self._load(file_name=FILENAME.DATA_FEATURES_Y_SCALED_TEST2, index_col=self.date_column_name)
        significant_features_df = self._load(FILENAME.SIGNIFICANT_FEATURES, None)
        significant_features = significant_features_df['Feature'].to_list()

        x_test1 = test1_data[significant_features]
        x_test2 = test2_data[significant_features]
        y_test1 = test1_data[self.target_column_name]
        y_test2 = test2_data[self.target_column_name]

        shap_values_1 = self._shap_beeswarm_analysis(model=best_model, x_test=x_test1, filename=FILENAME.SHAP_BEESWARM_TEST1)
        shap_values_2 = self._shap_beeswarm_analysis(model=best_model, x_test=x_test2, filename=FILENAME.SHAP_BEESWARM_TEST2)
        self._shap_waterfall_analysis(shap_values=shap_values_1, filename=FILENAME.SHAP_WATERFALL_TEST1, observation_count=5)
        self._shap_waterfall_analysis(shap_values=shap_values_2, filename=FILENAME.SHAP_WATERFALL_TEST2, observation_count=5)

        return data
    
    def _get_significant_features(self, data, param):
        default_importance_df = self._load(FILENAME.DEFAULT_FEATURE_IMPORTANCE, None)
        mod_importance_df = self._load(FILENAME.MODIFIED_FEATURE_IMPORTANCE, None)

        default_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
        mod_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

        print('default_importance_df')
        print(default_importance_df.head(10))
        print('mod_filtered_importance_df')
        print(mod_importance_df.head(10))

        threshold = param['feature_importance_threshold']
        filtered_importance_df = default_importance_df[default_importance_df['Importance'] > threshold]
        mod_filtered_importance_df = mod_importance_df[mod_importance_df['Importance'] > threshold]
        shared_features_df = pd.merge(filtered_importance_df, mod_filtered_importance_df, on = 'Feature', suffixes = ('_default', '_modified'))
        significant_features = shared_features_df['Feature'].to_list()
        print(f'# threshold: {threshold}, significant features ({len(significant_features)}): {significant_features}')
        

        train_data = self._load(FILENAME.DATA_FEATURES_Y_SCALED_TRAIN, self.date_column_name)
        train_data = train_data[significant_features + [self.target_column_name]]
        test1_data = self._load(FILENAME.DATA_FEATURES_Y_SCALED_TEST1, self.date_column_name)
        test1_data = test1_data[significant_features + [self.target_column_name]]
        test2_data = self._load(FILENAME.DATA_FEATURES_Y_SCALED_TEST2, self.date_column_name)
        test2_data = test2_data[significant_features + [self.target_column_name]]

        self._save(train_data, FILENAME.DATA_TRAIN_SF)
        self._save(test1_data, FILENAME.DATA_TEST1_SF)
        self._save(test2_data, FILENAME.DATA_TEST2_SF)
        self._save(shared_features_df, FILENAME.SIGNIFICANT_FEATURES)

        plot_feature_importance_comparison(first_model_features_df=default_importance_df,
                                second_model_features_df=mod_importance_df,
                                feature_col='Feature',
                                feature_importance_col='Importance',
                                model1_name='Deafult XGBoostModel',
                                model2_name='XGBoostModel With Parameter',
                                subset_count=30,
                                threshold=param['feature_importance_threshold'],
                                output_file_path=self._get_full_file_path(param['feature_importance_comparison_plot'])
                                )

        return data
    
    def get_strategy_result(self, best_model, test_data, significant_features, param):
        strategyManager = StrategyManager()
        predicted_prob = best_model.predict_proba(test_data[significant_features])
        predicted_signal = [x for x in predicted_prob[:, 1]]
        
        df = pd.DataFrame({
            'close': test_data['close'],
            'predicted_signal_prob': predicted_signal,
            'actual_signal': test_data[self.target_column_name]
        }, index=test_data.index)

        print(df.head(5))

        # # Perform the grid search
        results_df = strategyManager.grid_search_trade_setups(
            df,
            param['holding_periods'],
            param['take_profits'],
            param['stop_losses'],
            param['thresholds'],
            verbose=True
        )

        # # Print or save results
        print("Grid Search Results:")
        print(results_df.head(15))  # Show first 15 rows
        best_row = results_df.dropna(subset=['sharpe_ratio']).sort_values('sharpe_ratio', ascending=False).head(1)
        print("\nBest Combination by Sharpe Ratio:")
        print(best_row)
        return results_df, best_row

    def _run_strategy(self, data, param):
        best_model = self._load_model(self._get_full_file_path(FILENAME.BEST_XGBOOST_MODEL))
        test1_data = self._load(file_name=FILENAME.DATA_FEATURES_Y_SCALED_TEST1, index_col=self.date_column_name)
        test2_data = self._load(file_name=FILENAME.DATA_FEATURES_Y_SCALED_TEST2, index_col=self.date_column_name)
        significant_features_df = self._load(FILENAME.SIGNIFICANT_FEATURES, None)
        significant_features = significant_features_df['Feature'].to_list()

        results_df_test_1, best_row_test_1 = self.get_strategy_result(
            best_model=best_model,
            test_data=test1_data,
            significant_features=significant_features,
            param=param
        )
        results_df_test_2, best_row_test_2 = self.get_strategy_result(
            best_model=best_model,
            test_data=test2_data,
            significant_features=significant_features,
            param=param
        )
        self._save(data=results_df_test_1, file_name=FILENAME.STRATEGY_RESULT_TEST_1)
        self._save(data=results_df_test_2, file_name=FILENAME.STRATEGY_RESULT_TEST_2)
        self._save(best_row_test_1, FILENAME.BEST_STRATEGY_RESULT_TEST_1)
        self._save(best_row_test_2, FILENAME.BEST_STRATEGY_RESULT_TEST_2)
        return data
    
    def _get_best_model_params(self, data, param):
        grid_search_results_df = self._load(FILENAME.GRID_SEARCH_RESULTS, None)
        
        mlModelManager = MLModelManager()
        best_model_params = mlModelManager.get_best_model_params_and_plot(
            grid_search_results_df, 
            self._get_full_file_path(FILENAME.BEST_MODEL_PARAMS_PLOT)
        )
        print(f"Best Model params: {best_model_params}")

        data_train = self._load(file_name=FILENAME.DATA_FEATURES_Y_SCALED_TRAIN, index_col=self.date_column_name)
        significant_features_df = self._load(file_name=FILENAME.SIGNIFICANT_FEATURES, index_col=None)
        significant_features = significant_features_df['Feature'].to_list()

        x_train  = data_train[significant_features]
        y_train = data_train[self.target_column_name]

        print(x_train.head(5))
        print(y_train.head(5))
        
        # Train the model with the best parameters
        best_model = mlModelManager.train_model_with_best_params(x_train, y_train, best_model_params)
        self._save(pd.DataFrame(best_model_params), FILENAME.BEST_MODEL_PARAMS)

        # Output the trained model details (or use best_model for predictions, etc.)
        print("Trained XGBoost Model with Best Parameters:")
        print(f'best model: {best_model}')
        
        self._save_model(model=best_model, file_name=self._get_full_file_path(FILENAME.BEST_XGBOOST_MODEL))
        return data

    def _load_model(self, file_name):
        # Later, you can load the model with:
        with open(file_name, 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model
    
    def _save_model(self, model, file_name):
        # Save the model to a file
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)
    
    def _param_tuning_using_grid_search(self, data, param):
        data_train = self._load(file_name=FILENAME.DATA_FEATURES_Y_SCALED_TRAIN, index_col=self.date_column_name)
        test1_data = self._load(file_name=FILENAME.DATA_FEATURES_Y_SCALED_TEST1, index_col=self.date_column_name)
        test2_data = self._load(file_name=FILENAME.DATA_FEATURES_Y_SCALED_TEST2, index_col=self.date_column_name)

        significant_features_df = self._load(file_name=FILENAME.SIGNIFICANT_FEATURES, index_col=None)
        significant_features = significant_features_df['Feature'].to_list()

        x_train  = data_train[significant_features]
        y_train = data_train[self.target_column_name]

        x_test1 = test1_data[significant_features]
        y_test1 = test1_data[self.target_column_name]

        x_test2 = test2_data[significant_features]
        y_test2 = test2_data[self.target_column_name]


        print(f'x_train: {x_train.head(5)}')
        print(f'significant features: {significant_features_df.head(5)}')

        grid_search_result_df = param_tuning_using_grid_search(
            x_train=x_train, 
            y_train=y_train, 
            x_test1=x_test1, 
            y_test1=y_test1, 
            x_test2=x_test2, 
            y_test2=y_test2
        )

        self._save(grid_search_result_df, FILENAME.GRID_SEARCH_RESULTS)
        return data
    
    def _get_feature_importance(self, data, param):
        feature_info_df = self._load(FILENAME.FEATURE_INFO, None)
        feature_names = feature_info_df['feature_name'].to_list()
        default_importance_df, mod_importance_df = get_feature_improtance(
            data, 
            feature_cols = feature_names, 
            target_col = self.target_column_name, 
            feature_importance_threshold = param['feature_importance_threshold']
        )
        self._save(default_importance_df, FILENAME.DEFAULT_FEATURE_IMPORTANCE)
        self._save(mod_importance_df, FILENAME.MODIFIED_FEATURE_IMPORTANCE)
        return data
    
    def _split_data(self, data, param):
        training_data, test1_data, test2_data = time_train_test_split_3way(data, train_frac=0.7, test1_frac=0.15, test2_frac=0.15)
        self._save(training_data, FILENAME.DATA_FEATURES_Y_SCALED_TRAIN)
        self._save(test1_data, FILENAME.DATA_FEATURES_Y_SCALED_TEST1)
        self._save(test2_data, FILENAME.DATA_FEATURES_Y_SCALED_TEST2)
        
        return training_data

    def _scale_features(self, data, param):
        feature_info_df = self._load(FILENAME.FEATURE_INFO, None)
        binary_feature_names = feature_info_df[feature_info_df['feature_type'] == FeatureType.BINARY]['feature_name'].to_list()
        skip_columns = [self.target_column_name] + binary_feature_names
        print(f'binary features: {binary_feature_names}')
        print(f'skip columns for scaling: {skip_columns}')
        data, scaler = scale_features(data, skip_columns=skip_columns, method="minmax")
        print(data.head(5))
        return data
    
    def _generate_Y(self, data, param):
        feature_generator = FeatureGenerator()
        data = feature_generator._generate_target_Y_fast(data, future_periods=param['future_periods'], threshold = param['threshold'])
        print(f'target Y value counts: {data[self.target_column_name].value_counts()}')
        return data

    def run(self):
        data = None
        for i, func in enumerate(self._functions):
            print(f'{"="*self.line_border_len}')
            param = self.params[func.__name__]
            print(f'step-{i+1}: {param["description"]}')
            if param['run']:
                if param['load']:
                    data = self._load(param['load_file_name'], self.date_column_name)
                data = func(data, param)
                if not (data is None):
                    print(f'data size: {data.shape}')
                    print(data.head(5))
                if param['save']:
                    self._save(data, param['output_file_name'])
            print(f'{"="*self.line_border_len}')
    
    def _load_data(self, data, param):
        data = pd.read_csv(self.file_path, parse_dates=[self.date_column_name], index_col=self.date_column_name)
        return data

    def _exclude_observation(self, data, param):
        observation_exclusion = ObservationExclusion()
        data = observation_exclusion.exclude_data(data)
        print(f'After excluding data with day hour < {observation_exclusion._exclusion_logic_HOUR_LOW} am or > {observation_exclusion._exclusion_logic_HOUR_HIGH} pm, size: {data.size}')
        print(f'min timestamp: {data.index.min()}, max timestamp: {data.index.max()}')        
        return data
    
    def _generate_features(self, data, param):
        feature_generator = FeatureGenerator()
        data = feature_generator.generate(data)
        feature_names = feature_generator._feature_names
        feature_types = [feature_generator._feature_types[i] for i in feature_generator._feature_names]
        feature_info_df = pd.DataFrame({
            'feature_name': feature_names,
            'feature_type': feature_types
        })
        print(f'feature info: {feature_info_df}')
        print(f'Features: {feature_generator._feature_names}')
        self._feature_info_df = feature_info_df
        self._save(feature_info_df, param['feature_info_file_name'])
        return data
    
    def _save(self, data, file_name):
        filepath = os.path.join(self.file_suffix_dir, file_name)
        data.to_csv(filepath)
        print(f'# Saved to :{file_name}')
        print(f'Full file path: {filepath}')
    
    def _load(self, file_name, index_col):
        filepath = os.path.join(self.file_suffix_dir, file_name)
        print(f'# loaded data from :{file_name}')
        print(f'Full file path: {filepath}')
        return pd.read_csv(filepath, index_col=index_col)
    
'''
def load_data_plot_dist(file_suffix, is_google_colab=True, date_column_name='timestamp'):
    data_loader = DataLoader(is_google_colab=is_google_colab)
    file_dict = data_loader.load_data_from_drive_folder()
    file_path=file_dict[file_suffix]

    feature_generator = FeatureGenerator()
    line_border_len = 100


    file_suffix_dir = os.path.join(data_loader.intermediate_data_dir, file_suffix)
    if not os.path.exists(file_suffix_dir):
        os.makedirs(file_suffix_dir)


    
    # step-1: load data
    if RUN_SETUP.step_1['run']:
        print(f'{"="*line_border_len}')

        data = data_loader.load_data(file_path=file_path, date_column_name=date_column_name)
        print(f'{"="*line_border_len}')
        print(f'# step-1: Raw data:')
        print(f'Raw data size: {data.size}')
        print(data.head(5))
        print(f'{"="*line_border_len}')
    # plot_hourly_volume_distribution(data, percentiles_to_plot)

    # step-2: exclude observation.
    if RUN_SETUP.run_step_2:
        observation_exclusion = ObservationExclusion()
        data = observation_exclusion.exclude_data(data)
        print(f'{"="*line_border_len}')
        print(f'# step-2: Exclude Observation:')
        print(f'After excluding data with day hour < 10 am or > 3 pm, size: {data.size}')
        print(data.head(5))
        print(f'min timestamp: {data['timestamp'].min()}, max timestamp: {data['timestamp'].max()}')
        print(f'{"="*line_border_len}')
        if RUN_SETUP.save_intermediate_data:
            data.to_csv(os.path.join(file_suffix_dir, 'data_after_exclusion.csv'), index=False)

    # step-2.5: load already generated features
    
    if RUN_SETUP.run_step_3:
        print(f'{"="*line_border_len}')
        print(f'# step-2.5: Generate features:')
        if RUN_SETUP.load_intermediate_data:
            data = pd.read_csv(os.path.join(file_suffix_dir, 'data_after_exclusion.csv'))
            data.index = data['timestamp']

        data = feature_generator.generate(data)
        feature_names = feature_generator._feature_names
        feature_types = [feature_generator._feature_types[i] for i in feature_generator._feature_names]
        feature_info_df = pd.DataFrame({
                'feature_name': feature_names,
                'feature_type': feature_types
        })
        print(feature_names)
        print(feature_types)
        print(f'feature info: {feature_info_df}')
        
        print(f'Features: {feature_generator._feature_names}')
        print(data.head(5))
        if RUN_SETUP.save_intermediate_data:
            data.to_csv(os.path.join(file_suffix_dir, 'data_features.csv'), index=False)
            feature_info_df.to_csv(os.path.join(file_suffix_dir, 'feature_info.csv'), index=False)
        print(f'{"="*line_border_len}')

    # step-3: generate y features.
    if RUN_SETUP.run_step_4:
        print(f'{"="*line_border_len}')
        print(f'# step-3: Generate Y value:')
        if RUN_SETUP.load_intermediate_data:
            data = pd.read_csv(os.path.join(file_suffix_dir, 'data_features.csv'))
            feature_info_df = pd.read_csv(os.path.join(file_suffix_dir, 'feature_info.csv'))
            data.index = data['timestamp']

        data = feature_generator._generate_target_Y_fast(data, future_periods=120, threshold = 0.01)
        
        print(data.head(5))
        
        print(f'target Y value counts: {data['Y'].value_counts()}')

        # plot_Y_distribution_with_frequency(data[['Y']])
        
        if RUN_SETUP.save_intermediate_data:
            data.to_csv(os.path.join(file_suffix_dir, 'data_features_Y.csv'), index=False)
            
        print(f'{"="*line_border_len}')


    # step-5: scaling
    if RUN_SETUP.run_step_6:
        print(f'{"="*line_border_len}')
        print(f'# step-5: Scale features:')
        if RUN_SETUP.load_intermediate_data:
            data = pd.read_csv(os.path.join(file_suffix_dir, 'data_features_Y.csv'))
            feature_info_df = pd.read_csv(os.path.join(file_suffix_dir, 'feature_info.csv'))
            data.index = data['timestamp']
        
        binary_feature_names = feature_info_df[feature_info_df['feature_type'] == FeatureType.BINARY]['feature_name'].to_list()
        skip_columns = ['Y', 'timestamp'] + binary_feature_names
        print(f'binary features: {binary_feature_names}')
        print(f'skip columns for scaling: {skip_columns}')
        
        # Scale the data using StandardScaler
        data, scaler = scale_features(data, skip_columns=skip_columns, method="minmax")
        print(data.head(5))
        
        if RUN_SETUP.save_intermediate_data:
            data.to_csv(os.path.join(file_suffix_dir, 'data_features_Y_scaled.csv'), index=False)

        print(f'{"="*line_border_len}')

    # correlations = compute_feature_y_correlation(df_features, 'Y', feature_columns)
    # print("Correlation between each feature and Y variable:")
    # print(correlations)

    # step 6
    if RUN_SETUP.run_step_7:
        print(f'{"="*line_border_len}')
        print(f'# step-6: Split data into training, test1, test2:')
        if RUN_SETUP.load_intermediate_data:
            data = pd.read_csv(os.path.join(file_suffix_dir, 'data_features_Y_scaled.csv'))
            data.index = data['timestamp']
        training_data, test1_data, test2_data = time_train_test_split_3way(data, train_frac=0.70, test1_frac=0.15, test2_frac=0.15)
        # training_data, test1_data, test2_data = split_test_train_data(data)
        print(f'{training_data.shape}, {test1_data.shape}, {test2_data.shape}')
        if RUN_SETUP.save_intermediate_data:
            training_data.to_csv(os.path.join(file_suffix_dir, 'training_data.csv'), index=False)
            test1_data.to_csv(os.path.join(file_suffix_dir, 'test1_data.csv'), index=False)
            test2_data.to_csv(os.path.join(file_suffix_dir, 'test2_data.csv'), index=False)
        print(f'{"="*line_border_len}')

    # step-7
    if RUN_SETUP.run_step_8:
        print(f'{"="*line_border_len}')
        print(f'# step-7: Get feature importance:')
        if RUN_SETUP.load_intermediate_data:
            training_data = pd.read_csv(os.path.join(file_suffix_dir, 'training_data.csv'))
            training_data.index = training_data['timestamp']
            # test1_data = pd.read_csv(os.path.join(file_suffix_dir, 'test1_data.csv'))
            # test1_data.index = test1_data['timestamp']
            # test2_data = pd.read_csv(os.path.join(file_suffix_dir, 'test2_data.csv'))
            # test2_data.index = test2_data['timestamp']
            feature_info_df = pd.read_csv(os.path.join(file_suffix_dir, 'feature_info.csv'))
        feature_columns = feature_info_df['feature_name'].to_list()
        feature_importance_threshold=0.005
        default_feature_importance_df, mod_importance_df, shared_features_df = get_feature_improtance(training_data, feature_cols=feature_columns, target_col='Y', feature_importance_threshold=feature_importance_threshold)
        print(f'Import features for default xgboost model: {default_feature_importance_df}')
        print(f'Import features for modified xgboost model: {mod_importance_df}')
        print(f'Import features combined: {shared_features_df}')
        # Using the function to generate the same plot as before
        plot_feature_importance_comparison(first_model_features_df=default_feature_importance_df,
                                        second_model_features_df=mod_importance_df,
                                        feature_col='Feature',
                                        feature_importance_col='Importance',
                                        model1_name='Deafult XGBoostModel',
                                        model2_name='XGBoostModel With Parameter',
                                        subset_count=30,
                                        threshold=feature_importance_threshold)
        
        if RUN_SETUP.save_intermediate_data:
            default_feature_importance_df.to_csv(os.path.join(file_suffix_dir, 'default_feature_importance.csv'), index=False)
            mod_importance_df.to_csv(os.path.join(file_suffix_dir, 'mod_feature_importance.csv'), index=False)
            shared_features_df.to_csv(os.path.join(file_suffix_dir, 'shared_feature_importance.csv'), index=False)

        print(f'{"="*line_border_len}')

    # step-8: get x_train, y_train, x_test1, y_test1, x_test2, y_test2 for significant fetaure
    if RUN_SETUP.run_step_9:
        print(f'{"="*line_border_len}')
        print(f'# step-8: Get significant features:')
        if RUN_SETUP.load_intermediate_data:
            training_data = pd.read_csv(os.path.join(file_suffix_dir, 'training_data.csv'))
            training_data.index = training_data['timestamp']
            test1_data = pd.read_csv(os.path.join(file_suffix_dir, 'test1_data.csv'))
            test1_data.index = test1_data['timestamp']
            test2_data = pd.read_csv(os.path.join(file_suffix_dir, 'test2_data.csv'))
            test2_data.index = test2_data['timestamp']
            shared_features_df = pd.read_csv(os.path.join(file_suffix_dir, 'shared_feature_importance.csv'))
        significant_features = shared_features_df['Feature'].to_list()

        x_train_sf = training_data[significant_features]
        y_train_sf = training_data['Y']

        x_test1_sf = test1_data[significant_features]
        y_test1_sf = test1_data['Y']

        x_test2_sf = test2_data[significant_features]
        y_test2_sf = test2_data['Y']
        
        print(f'x_train_sf: {x_train_sf.shape}, y_train_sf: {y_train_sf.shape}, x_test1_sf: {x_test1_sf.shape}, y_test1_sf: {y_test1_sf.shape}, x_test2_sf: {x_test2_sf.shape}, y_test2_sf:{y_test2_sf.shape}')

        if RUN_SETUP.save_intermediate_data:
            x_train_sf.to_csv(os.path.join(file_suffix_dir, 'x_train_sf.csv'))
            y_train_sf.to_csv(os.path.join(file_suffix_dir, 'y_train_sf.csv'))
            x_test1_sf.to_csv(os.path.join(file_suffix_dir, 'x_test1_sf.csv'))
            y_test1_sf.to_csv(os.path.join(file_suffix_dir, 'y_test1_sf.csv'))
            x_test2_sf.to_csv(os.path.join(file_suffix_dir, 'x_test2_sf.csv'))
            y_test2_sf.to_csv(os.path.join(file_suffix_dir, 'y_test2_sf.csv'))
        print(f'{"="*line_border_len}')

    # step-9: grid search
    if RUN_SETUP.run_step_10:
        print(f'{"="*line_border_len}')
        print(f'# step-9: Grid search:')
        if RUN_SETUP.load_intermediate_data:
            x_train_sf = pd.read_csv(os.path.join(file_suffix_dir, 'x_train_sf.csv'), index_col='timestamp')
            y_train_sf = pd.read_csv(os.path.join(file_suffix_dir, 'y_train_sf.csv'), index_col='timestamp')
            x_test1_sf = pd.read_csv(os.path.join(file_suffix_dir, 'x_test1_sf.csv'), index_col='timestamp')
            y_test1_sf = pd.read_csv(os.path.join(file_suffix_dir, 'y_test1_sf.csv'), index_col='timestamp')
            x_test2_sf = pd.read_csv(os.path.join(file_suffix_dir, 'x_test2_sf.csv'), index_col='timestamp')
            y_test2_sf = pd.read_csv(os.path.join(file_suffix_dir, 'y_test2_sf.csv'), index_col='timestamp')

        result_df = param_tuning_using_grid_search(x_train_sf, y_train_sf, x_test1_sf, y_test1_sf, x_test2_sf, y_test2_sf)
        if RUN_SETUP.save_intermediate_data:
            result_df.to_csv(os.path.join(file_suffix_dir, 'grid_search_results.csv'), index=False)
        
        print(result_df.head(5))
        print(f'{"="*line_border_len}')

    # step-N: shap analysis, effect of feature on target Y column
    if RUN_SETUP.run_step_N:
        model, shap_vals = run_shap_analysis(data, feature_columns, 'Y', dependence_feature=feature_columns[0])
    # plot_hourly_volume_distribution(filtered_data, percentiles_to_plot)

'''
import sys 
if __name__ == "__main__":
    data_loader = DataLoader(is_google_colab=False)
    file_dict = data_loader.get_data_filepath_dict()
    for file_suffix in file_dict.keys():
    # for file_suffix in ['AMZN']:
        print(file_suffix)
        filepath = file_dict[file_suffix]
        print(f'Stock: {file_suffix}')
        RUN_SETUP(is_google_colab=False, file_suffix=file_suffix).run()
    # load_data_plot_dist(file_suffix='META', is_google_colab=False, date_column_name='timestamp')
