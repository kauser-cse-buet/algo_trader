# find the best model parameter
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import os

class MLModelManager:
    def choose_best_model(self, df):
        """
        Selects the best model/parameter combination from a DataFrame of AUC scores.

        The DataFrame must include the columns:
            - 'Trees'
            - 'Learning Rate'
            - 'Subsample'
            - '% Features'
            - 'Weight of Default'
            - 'AUC Train'
            - 'AUC Test 1'
            - 'AUC Test 2'

        The selection criteria are:
            1. Highest average test AUC = (AUC Test 1 + AUC Test 2) / 2
            2. Among any ties, smallest gap = AUC Train - average_test_auc

        Returns:
            pd.Series: The row corresponding to the best model configuration.
        """
        df = df.copy()
        # Compute average test AUC
        df['avg_test_auc'] = (df['AUC Test 1'] + df['AUC Test 2']) / 2
        # Compute the gap between train and average test AUC
        df['train_test_gap'] = df['AUC Train'] - df['avg_test_auc']
        # Sort: first by descending avg_test_auc, then ascending gap (lower gap is better)
        df_sorted = df.sort_values(by=['avg_test_auc', 'train_test_gap'], ascending=[False, True])
        best_model = df_sorted.iloc[0]
        return best_model

    def get_best_model_params_and_plot(self, df, plt_filepath = 'model_performance.png'):
        """
        Plots a scatter plot of average test AUC versus train-test gap for all model configurations,
        and marks & annotates the best configuration.

        The DataFrame must include:
            - 'Trees'
            - 'Learning Rate'
            - 'Subsample'
            - '% Features'
            - 'Weight of Default'
            - 'AUC Train'
            - 'AUC Test 1'
            - 'AUC Test 2'
        """
        df = df.copy()
        # Calculate average test AUC and gap
        df['avg_test_auc'] = (df['AUC Test 1'] + df['AUC Test 2']) / 2
        df['train_test_gap'] = df['AUC Train'] - df['avg_test_auc']

        # Choose the best model configuration
        best_model = self.choose_best_model(df)

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df['avg_test_auc'], df['train_test_gap'], c='blue', label='Model Configurations')

        # Mark the best model in red
        ax.scatter(best_model['avg_test_auc'], best_model['train_test_gap'], c='red', s=120, label='Best Model')

        ax.set_xlabel('Average Test AUC')
        ax.set_ylabel('Train-Test Gap (AUC Train - Avg Test AUC)')
        ax.set_title('Model Performance: Average Test AUC vs. Train-Test Gap')
        ax.legend()

        # Annotate the best model with its parameters
        annotation_text = (
            f"Trees: {best_model['Trees']}\n"
            f"LR: {best_model['Learning Rate']}\n"
            f"Subsample: {best_model['Subsample']}\n"
            f"% Features: {best_model['% Features']}\n"
            f"Weight: {best_model['Weight of Default']}\n"
            f"Avg Test AUC: {best_model['avg_test_auc']:.4f}"
        )
        ax.annotate(annotation_text,
                    xy=(best_model['avg_test_auc'], best_model['train_test_gap']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                    fontsize=9, ha='left')

        
        plt.savefig(plt_filepath)
        # plt.show()
        return best_model

    def train_model_with_best_params(self, X, y, best_params):
        """
        Trains an XGBoost classifier using the best model parameters.

        Parameters:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series or array-like): Target variable.
            best_params (dict or pd.Series): Dictionary/Series containing the best model parameters.
                Expected keys:
                    - 'Trees'            -> n_estimators
                    - 'Learning Rate'    -> learning_rate
                    - 'Subsample'        -> subsample
                    - '% Features'       -> colsample_bytree
                    - 'Weight of Default'-> scale_pos_weight (or other weight parameter)

        Returns:
            model (xgb.XGBClassifier): The trained XGBoost classifier.
        """
        # Map best_params to XGBoost parameter names and ensure they are of the correct type
        n_estimators = int(best_params['Trees'])  # Convert 'Trees' to integer
        learning_rate = float(best_params['Learning Rate']) # Convert 'Learning Rate' to float
        subsample = float(best_params['Subsample'])  # Convert 'Subsample' to float
        colsample_bytree = float(best_params['% Features']) # Convert '% Features' to float
        scale_pos_weight = float(best_params['Weight of Default']) # Convert 'Weight of Default' to float

        # Initialize XGBoost classifier with provided parameters.
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        # Train the model
        model.fit(X, y)
        self._model = model
        return model

from setup import DataLoader
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