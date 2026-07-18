from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, scale: float = 0.04):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * scale)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        max_src_len: int = 200,
        max_tgt_len: int = 50,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.src_embedding = nn.Linear(input_dim, d_model)
        self.tgt_embedding = nn.Linear(input_dim, d_model)
        self.src_pos = PositionalEncoding(d_model, max_src_len)
        self.tgt_pos = PositionalEncoding(d_model, max_tgt_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward, dropout, batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_decoder_layers)
        self.fc_out = nn.Linear(d_model, input_dim)

    def forward(self, src, tgt):
        src_emb = self.src_pos(self.src_embedding(src))
        tgt_emb = self.tgt_pos(self.tgt_embedding(tgt))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        memory = self.encoder(src_emb)
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.fc_out(out)

    def predict(self, src, steps: int = 1):
        self.eval()
        with torch.no_grad():
            tgt = src[:, -1:, :]
            outputs = []
            for _ in range(steps):
                pred = self.forward(src, tgt)
                next_step = pred[:, -1:, :]
                outputs.append(next_step)
                tgt = torch.cat([tgt, next_step], dim=1)
        return torch.cat(outputs, dim=1)


def train_transformer(
    model: TimeSeriesTransformer,
    train_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device | str = "cuda",
) -> list[float]:
    device = torch.device(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()
    epoch_losses: list[float] = []
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1, :]
            tgt_output = tgt[:, 1:, :]
            optimizer.zero_grad()
            output = model(src, tgt_input)
            loss = criterion(output, tgt_output)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)
        epoch_losses.append(avg_loss)
        scheduler.step()
    return epoch_losses
