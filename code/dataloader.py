from typing import Callable, Literal, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np


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

    def __init__(
        self,
        csv_path: str,
        datetime_column: str,
        target_column: str,
        context_length: int = 120,
        frequency: Literal["15min", "1h", "4h", "D"] = "1h",
        transform: Optional[Callable] = None,
    ):
        self.data = pd.read_csv(csv_path)
        self.data[datetime_column] = pd.to_datetime(self.data[datetime_column])
        self.data = self.data.groupby(pd.Grouper(key=datetime_column, freq=frequency)).sum().reset_index()
        self.data = self.data[[datetime_column, target_column]]
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)
        self.data["time_idx"] = self.data[datetime_column].dt.year * 12 + self.data[datetime_column].dt.month
        self.data["time_idx"] -= self.data["time_idx"].min()
        self.data = pd.get_dummies(self.data, columns=["month"])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {"data": self.data.iloc[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class CreateSequences(object):
    """Create sequences from a pandas DataFrame."""

    def __init__(self, context_length: int = 120):
        self.context_length = context_length

    def __call__(self, sample):
        data = sample["data"]
        data = data.drop(columns=["DateTime"])
        data = data.to_numpy()
        x = np.zeros((len(data) - self.context_length - self.target_length + 1, self.context_length, data.shape[1]))
        y = np.zeros((len(data) - self.context_length - self.target_length + 1, self.target_length, data.shape[1]))
        for i in range(len(data) - self.context_length - self.target_length + 1):
            x[i] = data[i : i + self.context_length]
            y[i] = data[i + self.context_length : i + self.context_length + self.target_length]
        return {"data": x, "target": y}


class ToTensor(object):
    """Convert pandas DataFrame to a Tensor."""

    def __init__(self, columns: list[str] = []):
        self.columns = columns

    def __call__(self, sample):
        data = sample["data"][self.columns] if self.columns else sample["data"]
        tensor_data = torch.from_numpy(data.values)
        return {"data": tensor_data}
