from typing import Literal
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklego.preprocessing import RepeatingBasisFunction
from sklearn.preprocessing import MinMaxScaler


class EliaSolarDataset(Dataset):
    """
    A PyTorch dataset class for loading and preprocessing Elia Solar dataset.

    Args:
        csv_path (str): The path to the CSV file containing the dataset.
        datetime_column (str): The name of the column containing the datetime information.
        target_column (str): The name of the column containing the target variable.
        context_length (int, optional): The length of the input context. Defaults to 120.
        frequency (Literal["15min", "1h", "4h", "D"], optional): The frequency of the data. Defaults to "1h".
        train_test_split_year (int, optional): The year to split the dataset into train and test sets. Defaults to 2021.

    Methods:
        get_dataframe(preprocessed: bool = False) -> pd.DataFrame:
            Returns the original or preprocessed dataframe.
    """

    def __init__(
        self,
        csv_path: str,
        datetime_column: str,
        target_column: str,
        context_length: int = 120,
        frequency: Literal["15min", "1h", "4h", "D"] = "1h",
        train_test_split_year: int = 2021,
        train_val_split_year: int = 2020,
    ):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.datetime_column = datetime_column
        self.target_column = target_column
        self.frequency = frequency
        self.data = self.__preprocess(self.data)

        self.train_test_split_index = (
            max(list(self.data[self.data[self.datetime_column].dt.year < train_test_split_year].index)) + context_length
        )

        self.train_val_split_index = (
            max(
                list(
                    self.data[
                        (self.data[self.datetime_column].dt.year >= train_val_split_year)
                        & (self.data[self.datetime_column].dt.year < train_test_split_year)
                    ].index
                )
            )
            + context_length
        )

        self.data = self.data.drop(columns=["DateTime"])
        self.data = torch.from_numpy(self.data.values.astype("float16"))
        self.data = self.data.unfold(0, context_length, 1)
        self.labels = self.data[:, -1, 0]

    def get_dataframe(self, preprocessed: bool = False) -> pd.DataFrame:
        """
        Returns the original dataframe, with or without preprocessing.

        Args:
            preprocessed (bool, optional): If True, returns the preprocessed dataframe. Defaults to False.

        Returns:
            pd.DataFrame: The original or preprocessed dataframe.
        """
        df = pd.read_csv(self.csv_path)
        if preprocessed:
            df = self.__preprocess(df)
        return df

    def __preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.datetime_column] = pd.to_datetime(df[self.datetime_column], format="%d/%m/%Y %H:%M")
        df = df.groupby(pd.Grouper(key=self.datetime_column, freq=self.frequency)).sum().reset_index()
        df = df[[self.datetime_column, self.target_column]]
        scaler = MinMaxScaler()
        df[self.target_column] = scaler.fit_transform(df[[self.target_column]]).flatten()
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {"data": self.data[idx], "label": self.labels[idx]}
        return sample
