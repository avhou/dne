import pandas as pd
from typing import Literal
from pathlib import Path
class DataLoader:

    def __init__(self, file_path: Path):
        """
        Initializes a DataLoader object.

        Args:
            file_path (Path): The path to the Excel file.
        """
        self.file_path = file_path

    def read_excel(self, aggregation_type: Literal["original", "daily"] = "original"):
        """
        Reads the Excel file and returns the data.

        Args:
            aggregation_type (Literal["original", "daily"], optional): The aggregation type of the data. Defaults to "Original".

        Returns:
            pandas.DataFrame: The data read from the Excel file.

        Raises:
            FileNotFoundError: If the file is not found.
            Exception: If any other error occurs during reading.
        """
        try:
            data = pd.read_excel(self.file_path, skiprows=3)
            data['DateTime'] = pd.to_datetime(data['DateTime'], format="%d/%m/%Y %H:%M")

            if aggregation_type == "daily":
                data = data.groupby(pd.Grouper(key='DateTime', freq='D')).sum().reset_index()

            return data
        except FileNotFoundError:
            print(f"File not found for path: ${self.file_path}.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

