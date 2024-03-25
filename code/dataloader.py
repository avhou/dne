import pandas as pd
from typing import Literal
from pathlib import Path

class DataLoader:

    def __init__(self, file_path: Path, aggregation_type: Literal["original", "daily"] = "original"):
        """
        Initializes a DataLoader object.

        Args:
            file_path (Path): The path to the Excel file.
            aggregation_type (Literal["original", "daily"], optional): The aggregation type of the data. Defaults to "Original".
        """
        self.file_path = file_path
        self.aggregation_type = aggregation_type

    def read_excel(self):
        """
        Reads the Excel file and returns the data.

        Returns:
            pandas.DataFrame: The data read from the Excel file.

        Raises:
            FileNotFoundError: If the file is not found.
            Exception: If any other error occurs during reading.
        """
        try:
            data = pd.read_excel(self.file_path, skiprows=3)

            if self.aggregation_type == "daily":
                data = data.groupby(pd.Grouper(key='datetime', freq='D')).sum().reset_index()

            return data
        except FileNotFoundError:
            print(f"File not found for path: ${self.file_path}.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

