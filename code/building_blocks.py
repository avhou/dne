# common classes
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import datetime
from datasets import EliaSolarDataset
from utils import ConfigSettings
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.fft import rfft


# %%
### Attention mechanism copied from Aladin Persson
### see https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/transformer_from_scratch
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float(-1e30) if energy.dtype == torch.float32 else -float(1e4))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return [out, out, out], attention


### Encodings
#### Standard Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


#### Temporal encoding
class TemporalEncoding(nn.Module):
    def __init__(self, d_model, frequency: Literal["15min", "1h", "d"] = "d"):
        super(TemporalEncoding, self).__init__()

        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        self.hour_embed = nn.Embedding(hour_size, d_model) if frequency in ["1h", "15min"] else 0
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.day_embed = nn.Embedding(day_size, d_model)
        self.month_embed = nn.Embedding(month_size, d_model)

    def forward(self, x):
        x = x.long()

        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x


#### Causal Conv1d encoding
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x):
        # Apply causal padding
        x = super(CausalConv1d, self).forward(F.pad(x, (self.__padding, 0)))
        x = F.tanh(x)
        return x


#### Asymmetric Padding Conv1d encoding
class AsymmetricConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_right, bias=True):
        super(AsymmetricConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
        self.padding_right = padding_right

    def forward(self, x):
        # Apply asymmetric padding
        x = F.pad(x, (0, self.padding_right))
        x = F.tanh(x)
        return self.conv(x)


### Encoder
#### Base
class TimeSeriesEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        kernel_size,
        padding_right,
    ):
        super(TimeSeriesEncoder, self).__init__()
        self.input_dim = input_dim
        self.embed_size = embed_size
        self.device = device
        self.dropout = nn.Dropout(dropout)

        self.kernel_size = kernel_size
        self.padding_right = padding_right

        self.temporal_embedding = TemporalEncoding(embed_size)
        self.pos_embedding = torch.nn.Embedding(embed_size, 512)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)
                for _ in range(num_layers)
            ]
        )

    def encoding(self, x):
        # Abstract method
        raise NotImplementedError("Encoding not implemented")

    def forward(self, x, mask):
        input_embedding = self.encoding(x)
        temporal_embed = self.temporal_embedding(x)

        input_embedding = [embed + temporal_embed for embed in input_embedding]

        attention_weights = []
        for layer in self.layers:
            input_embedding, weights = layer(*input_embedding, mask)
            attention_weights.append(weights)

        return input_embedding[0], attention_weights


#### Positional Encodig
class PositionalEncodingEncoder(TimeSeriesEncoder):
    def __init__(self, *args, **kwargs):
        super(PositionalEncodingEncoder, self).__init__(*args, **kwargs)
        self.feature_embedding = nn.Linear(self.input_dim, self.embed_size)
        self.pos_embedding = PositionalEncoding(self.embed_size)

    def encoding(self, x):
        feature_embed = self.feature_embedding(x.squeeze())
        feature_embed = feature_embed.unsqueeze(1).expand(-1, self.input_dim, -1)
        # print(f"feature_embed shape:{feature_embed.shape}")
        pos_embed = self.pos_embedding(x)
        # print(f"pos_embed shape:{pos_embed.shape}")

        input_embedding = self.dropout(feature_embed + pos_embed)
        return [input_embedding, input_embedding, input_embedding]


class FourierEncoder(PositionalEncodingEncoder):
    def __init__(self, *args, **kwargs):
        super(PositionalEncodingEncoder, self).__init__(*args, **kwargs)

    def encoding(self, x: torch.Tensor):
        print(f"fourier encoding shape:{x.shape}")
        x = x.permute(0, 2, 1)  # Permute dimensions to (batch_size, sequence_length, input_dim)
        print(f"fourier encoding shape:{x.shape}")
        x = x.flatten(start_dim=1)  # Flatten the tensor along the sequence_length dimension
        print(f"fourier encoding shape:{x.shape}")
        x = rfft(x.numpy())  # Apply the fast Fourier transform
        x = torch.from_numpy(x)
        print(f"fourier encoding shape:{x.shape}")
        x = x.unsqueeze(1)  # Add a singleton dimension for compatibility with other encodings
        print(f"fourier encoding shape:{x.shape}")

        return super(PositionalEncodingEncoder, self).encoding(x)


#### Causal Conv Encodig
class CausalConv1dEncoder(TimeSeriesEncoder):
    def __init__(self, *args, **kwargs):
        super(CausalConv1dEncoder, self).__init__(*args, **kwargs)
        self.qk_feature_embedding = CausalConv1d(self.input_dim, self.embed_size, self.kernel_size)
        self.v_feature_embedding = CausalConv1d(self.input_dim, self.embed_size, 1)

    def encoding(self, x):
        query_key_embed = self.qk_feature_embedding(x.permute(0, 2, 1)).permute(2, 0, 1)
        value_embed = self.v_feature_embedding(x.permute(0, 2, 1)).permute(2, 0, 1)

        pos_embed = self.pos_embedding(x.type(torch.long).squeeze()).permute(1, 0, 2)

        query_key_embed = query_key_embed + pos_embed
        value_embed = value_embed + pos_embed
        return [value_embed.permute(1, 0, 2), query_key_embed.permute(1, 0, 2), query_key_embed.permute(1, 0, 2)]


#### Asym Padding Conv Encodig
class AsymPaddingConv1dEncoder(TimeSeriesEncoder):
    # Tested with kernel_size=3, padding_right=2
    def __init__(self, *args, **kwargs):
        super(AsymPaddingConv1dEncoder, self).__init__(*args, **kwargs)
        self.qk_feature_embedding = AsymmetricConv1d(
            self.input_dim, self.embed_size, self.kernel_size, self.padding_right
        )
        self.v_feature_embedding = AsymmetricConv1d(self.input_dim, self.embed_size, 1, 0)

    def encoding(self, x):
        query_key_embed = self.qk_feature_embedding(x.permute(0, 2, 1)).permute(2, 0, 1)
        value_embed = self.v_feature_embedding(x.permute(0, 2, 1)).permute(2, 0, 1)

        pos_embed = self.pos_embedding(x.type(torch.long).squeeze()).permute(1, 0, 2)

        query_key_embed = query_key_embed + pos_embed
        value_embed = value_embed + pos_embed
        return [value_embed.permute(1, 0, 2), query_key_embed.permute(1, 0, 2), query_key_embed.permute(1, 0, 2)]


### Decoder
class TimeSeriesLinearDecoder(nn.Module):
    def __init__(self, embed_size, forecast_size=1):
        super(TimeSeriesLinearDecoder, self).__init__()

        self.embed_size = embed_size
        self.forecast_size = forecast_size

        self.decoder = nn.Linear(embed_size, forecast_size)

    def forward(self, x):
        x = self.decoder(x[:, -1, :])
        return x


@dataclass
class TimeSeriesTransformerParams:
    input_dim: int
    embed_size: int
    num_layers: int
    heads: int
    device: str
    forward_expansion: int
    dropout: float
    forecast_size: int
    encoder_type: str
    kernel_size: int
    padding_right: int


### Transformer
class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        forecast_size=1,
        encoder_type="PositionalEncodingEncoder",
        kernel_size=9,
        padding_right=2,
    ):
        super(TimeSeriesTransformer, self).__init__()

        encoder_mapping = {
            "PositionalEncodingEncoder": PositionalEncodingEncoder,
            "CausalConv1dEncoder": CausalConv1dEncoder,
            "AsymPaddingConv1dEncoder": AsymPaddingConv1dEncoder,
            "FourierEncoder": FourierEncoder,
        }

        Encoder = encoder_mapping.get(encoder_type, None)

        if Encoder is None:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.encoder = Encoder(
            input_dim, embed_size, num_layers, heads, device, forward_expansion, dropout, kernel_size, padding_right
        )

        self.decoder = TimeSeriesLinearDecoder(embed_size, forecast_size)

    def make_mask(self, x):
        seq_length = x.shape[1]
        return torch.triu(torch.ones(seq_length, seq_length) * float("-inf"), diagonal=1).to(x.device)

    def forward(self, x):
        mask = self.make_mask(x)
        encoded, attention_layers = self.encoder(x, mask)
        decoder = self.decoder(encoded)

        return decoder, attention_layers

    def from_params(params: TimeSeriesTransformerParams):
        return TimeSeriesTransformer(
            input_dim=params.input_dim,
            embed_size=params.embed_size,
            num_layers=params.num_layers,
            heads=params.heads,
            device=params.device,
            forward_expansion=params.forward_expansion,
            dropout=params.dropout,
            forecast_size=params.forecast_size,
            encoder_type=params.encoder_type,
            kernel_size=params.kernel_size,
            padding_right=params.padding_right,
        )


@dataclass
class ScenarioParams:
    name: str
    device: str
    epochs: int
    dataloader_train: Any
    dataloader_validation: Any
    dataloader_test: Any
    base_path: str
    early_stop_count: int = 5


@dataclass
class ScenarioResult:
    validation_losses: List[float]
    train_losses: List[float]
    test_losses: List[float]


@dataclass
class Scenario:
    params: ScenarioParams

    def train_one_epoch(self, model, train_loader, device, optimizer, criterion, scaler):
        model.train()
        train_loss_batch = []
        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs, attention = model(x_batch)
                    loss = criterion(outputs, y_batch)
            else:
                outputs, attention = model(x_batch)
                loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss_batch.append(loss.item())

        avg_train_loss = np.mean(train_loss_batch)
        return avg_train_loss

    def validate(self, model, val_loader, device, criterion):
        model.eval()
        val_loss_batch = []
        with torch.no_grad():
            for batch in val_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs, attention = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss_batch.append(loss.item())

        avg_val_loss = np.mean(val_loss_batch)
        return avg_val_loss

    def save_model_state(self, model, suffix: str):
        weights_dir = os.path.join(self.params.base_path, "weights")
        path = os.path.join(weights_dir, f"{self.params.name}_model_{suffix}.pth")
        torch.save(model.state_dict(), path)
        print(f"Model {self.params.name} saved to file {path} with suffix {suffix}")

    def execute(self, model) -> ScenarioResult:
        min_train_loss = float("inf")
        min_val_loss = float("inf")
        early_stop_count = 0
        train_losses = []
        val_losses = []
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5, verbose=True)
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        print(f"checking paths, base path is {self.params.base_path}")
        weights_dir = os.path.join(self.params.base_path, "weights")
        if not os.path.exists(weights_dir):
            print(f"creating directory {weights_dir}")
            os.makedirs(weights_dir, exist_ok=True)

        print(f"training and validating the model")
        for epoch in range(self.params.epochs):
            avg_train_loss = self.train_one_epoch(
                model, self.params.dataloader_train, self.params.device, optimizer, criterion, scaler
            )
            train_losses.append(avg_train_loss)

            if avg_train_loss < min_train_loss:
                min_train_loss = avg_train_loss
                print(f"New best training score at epoch {epoch+1}")
                self.save_model_state(model, f"best_train")

            avg_val_loss = self.validate(model, self.params.dataloader_validation, self.params.device, criterion)
            val_losses.append(avg_val_loss)

            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                print(f"New best validation score at epoch {epoch+1}")
                self.save_model_state(model, f"best_validation")

            scheduler.step(avg_val_loss)

            self.save_model_state(model, f"last_epoch")

            print(
                f"{datetime.datetime.now()}: scenario {self.params.name} epoch {epoch + 1}/{self.params.epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}"
            )
            if avg_val_loss > min_val_loss:
                print(f"increasing early stop count")
                early_stop_count += 1
                if early_stop_count >= self.params.early_stop_count:
                    print("Early stopping!")
                    break
            else:
                early_stop_count = 0

        print(f"testing the model")
        model.eval()
        test_losses = []
        with torch.no_grad():
            for x_batch, y_batch in self.params.dataloader_test:
                outputs, attention = model(x_batch.to(self.params.device))
                loss = criterion(outputs, y_batch.to(self.params.device))
                test_losses.append(loss.item())

        print(f"saving all losses to disk")
        path = os.path.join(weights_dir, f"{self.params.name}_train_val_losses.pkl")
        pd.DataFrame({"training": train_losses, "validation": val_losses}).to_pickle(path)
        path = os.path.join(weights_dir, f"{self.params.name}_test_losses.pkl")
        pd.DataFrame({"test": test_losses}).to_pickle(path)

        return ScenarioResult(val_losses, train_losses, test_losses)


def to_sequences(seq_size, obs):
    x = []
    y = []
    for i in range(len(obs) - seq_size):
        window = obs[i : (i + seq_size)]
        after_window = obs[i + seq_size]
        x.append(window)
        y.append(after_window)
    return torch.tensor(x, dtype=torch.float32).view(-1, seq_size, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)


def filename_part(encoder_type: str, frequency: str, layer: int, head: int, forward_expansion: int) -> str:
    return f"enc{encoder_type}-freq{frequency}-layers{layer}-heads{head}-fe{forward_expansion}"


def generate_dataset(cf: ConfigSettings, frequency: Literal["15min", "1h", "4h", "D"]) -> EliaSolarDataset:
    return EliaSolarDataset(
        csv_path=cf.data.file_path,
        datetime_column="DateTime",
        target_column="Corrected Upscaled Measurement [MW]",
        context_length=cf.model.context_length,
        frequency=frequency,
        train_test_split_year=cf.data.train_test_split_year,
        train_val_split_year=cf.data.train_val_split_year,
    )


def generate_model_params(
    cf: ConfigSettings, device: str, encoder_type: str, num_layer: int, num_head: int, forward_expansion: int
) -> TimeSeriesTransformerParams:
    return TimeSeriesTransformerParams(
        input_dim=cf.model.context_length,
        embed_size=cf.model.embedding_size,
        num_layers=num_layer,
        heads=num_head,
        device=device,
        forward_expansion=forward_expansion,
        dropout=cf.model.dropout,
        forecast_size=cf.model.forecast_size,
        encoder_type=encoder_type,
        kernel_size=cf.model.kernel_size,
        padding_right=cf.model.padding_right,
    )


def generate_loaders(
    cf: ConfigSettings, frequency: Literal["15min", "1h", "4h", "D"]
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # setup the dataset, once per frequency
    solar_dataset = generate_dataset(cf, frequency)

    # setup the dataloaders, once per frequency
    indices = list(range(len(solar_dataset)))
    train_indices = indices[: solar_dataset.train_val_split_index]
    val_indices = indices[solar_dataset.train_val_split_index : solar_dataset.train_test_split_index]
    test_indices = indices[solar_dataset.train_test_split_index :]

    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(solar_dataset, batch_size=cf.model.batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(
        solar_dataset, batch_size=cf.model.batch_size, sampler=valid_sampler
    )
    test_loader = torch.utils.data.DataLoader(solar_dataset, batch_size=cf.model.batch_size, sampler=test_sampler)
    return (train_loader, validation_loader, test_loader)


def generate_scenarios(
    base_name: str,
    device: str,
    encoder_type: str,
    frequencies: List[Literal["15min", "1h", "4h", "D"]],
    layers: List[int],
    heads: List[int],
    forward_expansions: List[int],
    base_path: str = "./",
) -> List[Tuple[TimeSeriesTransformerParams, ScenarioParams]]:
    cf = ConfigSettings(config_path="config.ini")
    params = []
    for frequency in frequencies:
        (train_loader, validation_loader, test_loader) = generate_loaders(cf, frequency)

        for num_layer in layers:
            for num_head in heads:
                for forward_expansion in forward_expansions:
                    model_params = generate_model_params(
                        cf, device, encoder_type, num_layer, num_head, forward_expansion
                    )
                    scenario_params = ScenarioParams(
                        name=f"elia-{base_name}-freq{frequency}-layers{num_layer}-heads{num_head}-fe{forward_expansion}",
                        device=device,
                        epochs=100,
                        dataloader_train=train_loader,
                        dataloader_validation=validation_loader,
                        dataloader_test=test_loader,
                        base_path="/dne" if cf.runtime.run_in_colab else base_path,
                    )
                    params.append((model_params, scenario_params))
    return params
