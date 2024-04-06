# common classes
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os


#%%
### Attention mechanism copied from Aladin Persson
### see https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/transformer_from_scratch
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

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
            energy = energy.masked_fill(mask == 0,  float(-1e+30) if energy.dtype == torch.float32 else -float(1e+4))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
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
        return out, attention


### Standard Positional encoding
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
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


### Encoder
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
    ):

        super(TimeSeriesEncoder, self).__init__()

        self.input_dim = input_dim
        self.embed_size = embed_size
        self.device = device

        self.feature_embedding = nn.Linear(input_dim, embed_size)
        self.pos_embedding = PositionalEncoding(embed_size)

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask):
        #print(f"input shape:{x.shape}")
        feature_embed = self.dropout(self.feature_embedding(x.squeeze()))
        feature_embed = feature_embed.unsqueeze(1).expand(-1, self.input_dim, -1)
        #print(f"feature_embed shape:{feature_embed.shape}")
        pos_embed = self.dropout(self.pos_embedding(x))
        #print(f"pos_embed shape:{pos_embed.shape}")

        input_embedding = feature_embed + pos_embed
        attention_layers = []

        for layer in self.layers:
            input_embedding, attention = layer(input_embedding, input_embedding, input_embedding, mask)
            attention_layers.append(attention)

        return input_embedding, attention_layers


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
            forecast_size=1
    ):
        super(TimeSeriesTransformer, self).__init__()

        self.encoder = TimeSeriesEncoder(
            input_dim,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout
        )

        self.decoder = TimeSeriesLinearDecoder(
            embed_size,
            forecast_size
        )

    def make_mask(self, x):
        seq_length = x.shape[1]
        return torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1).to(x.device)

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
           forecast_size=params.forecast_size
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


    def execute(self, model) -> ScenarioResult:
        min_train_loss = float('inf')
        min_val_loss = float('inf')
        best_model_state = None
        early_stop_count = 0
        train_losses = []
        val_losses = []
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        print(f"checking paths, base path is {self.params.base_path}")
        weights_dir = os.path.join(self.params.base_path, "weights")
        if not os.path.exists(weights_dir):
            print(f"creating directory {weights_dir}")
            os.makedirs(weights_dir, exist_ok=True)

        for epoch in range(self.params.epochs):
            avg_train_loss = self.train_one_epoch(model, self.params.dataloader_train, self.params.device, optimizer, criterion, scaler)
            train_losses.append(avg_train_loss)

            if avg_train_loss < min_train_loss:
                min_train_loss = avg_train_loss
                best_model_state = {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "type": "train"
                }
                print(f"New best training score at epoch {epoch+1}")

            avg_val_loss = self.validate(model, self.params.dataloader_validation, self.params.device, criterion)
            val_losses.append(avg_val_loss)

            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                best_model_state = {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "type": "val"
                }
                print(f"New best validation score at epoch {epoch+1}")

            scheduler.step(avg_val_loss)

            if avg_val_loss >= min_val_loss:
                early_stop_count += 1
                if early_stop_count >= self.params.early_stop_count:
                    print("Early stopping!")
                    break
            else:
                early_stop_count = 0

            print(f"Epoch {epoch + 1}/{self.params.epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

        if best_model_state:
            model_type = "train" if best_model_state["type"] == "train" else "val"
            path = os.path.join(weights_dir, f'{self.params.name}_model_best_{model_type}.pth')
            torch.save(best_model_state["state_dict"], path)
            print(f"Best {model_type} model saved to file {path} from epoch {best_model_state['epoch']+1}")

        return ScenarioResult(val_losses, train_losses)

def to_sequences(seq_size, obs):
    x = []
    y = []
    for i in range(len(obs) - seq_size):
        window = obs[i:(i + seq_size)]
        after_window = obs[i + seq_size]
        x.append(window)
        y.append(after_window)
    return torch.tensor(x, dtype=torch.float32).view(-1, seq_size,1), torch.tensor(y, dtype=torch.float32).view(-1, 1)


