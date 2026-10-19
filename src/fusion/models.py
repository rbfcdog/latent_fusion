import torch
from torch import nn

D_TS = 1024
D_TEXT = 2 * 384 + 1


class NumericEncoder(nn.Module):
    def __init__(self, d_ts=D_TS, d_hidden=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_ts, d_hidden), nn.LayerNorm(d_hidden))
        self.attn = nn.MultiheadAttention(d_hidden, n_heads, dropout=dropout, batch_first=True)

    def forward(self, z_ts):
        x = self.proj(z_ts)
        context, _ = self.attn(x[:, -1:], x, x, need_weights=False)
        return x[:, -1] + context[:, 0]


class TSOnlyPredictor(nn.Module):
    def __init__(self, d_ts=D_TS, d_hidden=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.numeric = NumericEncoder(d_ts, d_hidden, n_heads, dropout)
        self.head = nn.Sequential(nn.Linear(d_hidden + 1, d_hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d_hidden, 1))

    def forward(self, z_ts, z_text, text_mask, past_vol, return_attn=False):
        numeric = self.numeric(z_ts)
        known_vol = torch.log(past_vol.clamp_min(1e-6)).unsqueeze(-1)
        return self.head(torch.cat([numeric, known_vol], dim=-1)), {}


class TextOnlyPredictor(nn.Module):
    def __init__(self, d_text=D_TEXT, d_hidden=64, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_text, d_hidden), nn.LayerNorm(d_hidden))
        self.head = nn.Sequential(nn.Linear(d_hidden + 1, d_hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d_hidden, 1))

    def forward(self, z_ts, z_text, text_mask, past_vol, return_attn=False):
        mask = text_mask.unsqueeze(-1)
        text = self.proj(z_text)
        pooled = (text * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        coverage = text_mask.float().mean(dim=1, keepdim=True)
        prediction = self.head(torch.cat([pooled, coverage], dim=-1))
        fallback = torch.log(past_vol.clamp_min(1e-6)).unsqueeze(-1)
        return torch.where(text_mask.any(dim=1, keepdim=True), prediction, fallback), {}


class FusionPredictor(TSOnlyPredictor):
    def __init__(self, d_ts=D_TS, d_text=D_TEXT, d_hidden=64, n_heads=4, dropout=0.1):
        super().__init__(d_ts, d_hidden, n_heads, dropout)
        self.text_proj = nn.Sequential(nn.Linear(d_text, d_hidden), nn.LayerNorm(d_hidden))
        self.null_news = nn.Parameter(torch.zeros(1, 1, d_hidden))
        self.cross_attn = nn.MultiheadAttention(d_hidden, n_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(2 * d_hidden + 2, d_hidden)

    def forward(self, z_ts, z_text, text_mask, past_vol, return_attn=False):
        numeric = self.numeric(z_ts)
        text = self.text_proj(z_text)
        keys = torch.cat([text, self.null_news.expand(len(text), -1, -1)], dim=1)
        padding = torch.cat([~text_mask, torch.zeros((len(text), 1), dtype=torch.bool, device=text.device)], dim=1)
        context, weights = self.cross_attn(numeric.unsqueeze(1), keys, keys,
                                           key_padding_mask=padding, need_weights=return_attn)
        known_vol = torch.log(past_vol.clamp_min(1e-6)).unsqueeze(-1)
        coverage = text_mask.float().mean(dim=1, keepdim=True)
        gate = torch.sigmoid(self.gate(torch.cat([numeric, context[:, 0], known_vol, coverage], dim=-1)))
        gated = numeric + gate * context[:, 0] * text_mask.any(dim=1, keepdim=True)
        return self.head(torch.cat([gated, known_vol], dim=-1)), {"attn_ts_to_news": weights} if return_attn else {}


def build_model(name, d_ts=D_TS, d_text=D_TEXT):
    if name == "ts_only":
        return TSOnlyPredictor(d_ts)
    if name == "text_only":
        return TextOnlyPredictor(d_text)
    if name == "fusion":
        return FusionPredictor(d_ts, d_text)
    raise ValueError(f"modelo desconhecido: {name}")
