# prompt: create a function load data from google drive folder.  for each file name, store in a dictionary with first part of file name before _. return the dictionary

import os
import pandas as pd
is_google_colab = False
import sys 
class DataLoader:
    def __init__(self, is_google_colab=False):
        self.is_google_colab = is_google_colab
        if is_google_colab:
            from google.colab import drive
            drive.mount('/content/drive')
            self.project_dir = '/content/drive/My Drive/project_algo_trading'
            self.data_dir = os.path.join(self.project_dir, 'data')
            self.intermediate_data_dir = os.path.join(self.project_dir, 'data', 'intermediate')
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Move up two levels to reach the project directory
            project_dir = os.path.dirname(os.path.dirname(current_dir))
            # Print to verify
            # print("Current Directory:", current_dir)
            # print("Project Directory:", project_dir)
            self.project_dir = project_dir
            self.data_dir = os.path.join(self.project_dir, 'data', "historical_intraday", "1min_data")
            self.intermediate_data_dir = os.path.join(self.project_dir, 'data', 'intermediate')
        if not os.path.exists(self.intermediate_data_dir):
            os.makedirs(self.intermediate_data_dir)

    # def load_data(self, file_path, date_column_name='Date'):
    #     """
    #     Load raw BIST 100 data from a CSV file.
    #     The CSV is expected to have columns: Date, Open, High, Low, Close, Volume.
    #     """
    #     data = pd.read_csv(file_path, parse_dates=[date_column_name], index_col=date_column_name)
    #     return data

    def get_data_filepath_dict(self, folder_path=None):
        """Loads data from a Google Drive folder and stores filenames in a dictionary.

        Args:
        folder_path: The path to the folder in Google Drive.

        Returns:
        A dictionary where keys are the first part of filenames (before "_")
        and values are the full filenames. Returns an empty dictionary if the folder
        does not exist or if no files are found.
        """
        full_folder_path = None
        if folder_path:
            full_folder_path = os.path.join(self.data_dir, folder_path)
        else:
            full_folder_path = self.data_dir

        file_dict = {}
        if os.path.exists(full_folder_path):
            for filename in os.listdir(full_folder_path):
                if "_" in filename:
                    key = filename.split("_")[0]
                    full_filepath = os.path.join(full_folder_path, filename)
                    file_dict[key] = full_filepath
                else:
                    # Handle cases where the "_" is absent
                    full_filepath = os.path.join(full_folder_path, filename)
                    file_dict[filename] = full_filepath
        else:
            print(f"Error: Folder '{folder_path}' not found in Google Drive.")

        return file_dict

if "__name__" == "__main__":
    data_loader = DataLoader(is_google_colab = False)
    file_dict = data_loader.get_data_filepath_dict()
    print(file_dict.keys())