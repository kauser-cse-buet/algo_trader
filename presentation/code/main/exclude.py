import pandas as pd

class ObservationExclusion:
    def __init__(self):
        self._exclusion_logic_HOUR_LOW = 10
        self._exclusion_logic_HOUR_HIGH = 15
        self.exclusion_functions = [
            self._exclusion_logic_HOUR
        ]

    def exclude_data(self, df):
        """
        Excludes data from a DataFrame based on the provided exclusion logic.

        Args:
            data: The input DataFrame with a 'timestamp' column.
            exclusion_logic: A function that takes a DataFrame row as input and returns True if the row should be excluded, False otherwise.

        Returns:
            A new DataFrame with the excluded rows removed.
        """
        data = df.copy()
        data['timestamp'] = df.index
        # Ensure the timestamp column is of datetime type
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])

        data['hour'] = data['timestamp'].dt.hour

        # Apply the exclusion logic to filter the DataFrame
        for exclusion_logic in self.exclusion_functions:
            filtered_data = data[~data.apply(exclusion_logic, axis=1)]

        # Drop the temporary 'hour' column if it exists
        if 'hour' in filtered_data.columns:
            filtered_data = filtered_data.drop(['hour', 'timestamp'], axis=1)

        return filtered_data

    # Example usage:
    def _exclusion_logic_HOUR(self, row):
        return row['hour'] < self._exclusion_logic_HOUR_LOW or row['hour'] > self._exclusion_logic_HOUR_HIGH