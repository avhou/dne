import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import building_blocks
from building_blocks import *

from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets import EliaSolarDataset
from utils import ConfigSettings

cf = ConfigSettings(config_path="config.ini")

solar_dataset = EliaSolarDataset(
    csv_path=cf.data.file_path,
    datetime_column="DateTime",
    target_column="Corrected Upscaled Measurement [MW]",
    context_length=cf.model.context_length,
    frequency=cf.data.frequency,
    train_test_split_year=cf.data.train_test_split_year,
    train_val_split_year=cf.data.train_val_split_year,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

indices = list(range(len(solar_dataset)))
train_indices = indices[: solar_dataset.train_val_split_index]
val_indices = indices[solar_dataset.train_val_split_index : solar_dataset.train_test_split_index]
test_indices = indices[solar_dataset.train_test_split_index :]

# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(solar_dataset, batch_size=cf.model.batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(solar_dataset, batch_size=cf.model.batch_size, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(solar_dataset, batch_size=cf.model.batch_size, sampler=test_sampler)

print(cf.model.encoder_type)

# https://stackoverflow.com/questions/65996797/how-to-refresh-a-python-import-in-a-jupyter-notebook-cell
from importlib import reload

reload(building_blocks)
from building_blocks import *

model_params = TimeSeriesTransformerParams(
    input_dim=cf.model.context_length,
    embed_size=cf.model.embedding_size,
    num_layers=cf.model.num_layers,
    heads=cf.model.num_attention_heads,
    device=device,
    forward_expansion=cf.model.forward_expansion,
    dropout=cf.model.dropout,
    forecast_size=cf.model.forecast_size,
    encoder_type=cf.model.encoder_type,
    kernel_size=cf.model.kernel_size,
    padding_right=cf.model.padding_right,
)
scenario_params = ScenarioParams(
    name="electricity",
    device=device,
    epochs=100,
    dataloader_train=train_loader,
    dataloader_validation=validation_loader,
    dataloader_test=test_loader,
    base_path="/dne" if cf.runtime.run_in_colab else "./",
)
model = TimeSeriesTransformer.from_params(model_params).to(device)
scenario = Scenario(scenario_params)
result = scenario.execute(model)
print(f"execution done")
