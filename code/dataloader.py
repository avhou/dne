from typing import Literal
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class EliaSolarDataset(Dataset):
    """
    A custom dataset class for loading and processing Elia solar data.

    Args:
        csv_path (str): The path to the CSV file containing the data.
        datetime_column (str): The name of the column containing the datetime information.
        frequency (Literal["15min", "1h", "4h", "D"], optional): The frequency at which to group the data. Defaults to "15min".

    Attributes:
        data (pd.DataFrame): The loaded and processed data.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns a specific sample from the dataset.

    """

    def __init__(self, csv_path: str, datetime_column: str, frequency: Literal["15min", "1h", "4h", "D"] = "15min"):
        self.data = pd.read_csv(csv_path)
        self.data[datetime_column] = pd.to_datetime(self.data[datetime_column])
        self.data = self.data.groupby(pd.Grouper(key=datetime_column, freq=frequency)).sum().reset_index()
        self.data["time_idx"] = self.data[datetime_column].dt.year * 12 + self.data[datetime_column].dt.month
        self.data["time_idx"] -= self.data["time_idx"].min()
        self.data["month"] = self.data.date.dt.month.astype(str).astype("category")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        return sample


class StandardScaleTransform(object):
    """Apply standard scaling to selected columns of a pandas DataFrame."""

    def __init__(self, columns: list[str]):
        self.columns = columns
        self.scaler = StandardScaler()

    def __call__(self, sample):
        data = sample["data"]
        transformed_data = self.scaler.fit_transform(data)
        return {"data": transformed_data}


class ToTensor(object):
    """Convert pandas DataFrame to a Tensor."""

    def __init__(self, columns: list[str] = []):
        self.columns = columns

    def __call__(self, sample):
        data = sample["data"][self.columns] if self.columns else sample["data"]
        tensor_data = torch.from_numpy(data.values)
        return {"data": tensor_data}
