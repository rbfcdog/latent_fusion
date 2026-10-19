import torch
import torch.nn as nn

from models.temporal_fusion import TemporalFusion

D_TS = 1024
D_TEXT = 384


class UnimodalPredictor(nn.Module):
    def __init__(self, d_in, d_hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, z_ts, z_llm):
        raise NotImplementedError


class TSOnlyPredictor(UnimodalPredictor):
    def forward(self, z_ts, z_llm):
        return self.net(z_ts.mean(dim=1)), {}


class TextOnlyPredictor(UnimodalPredictor):
    def forward(self, z_ts, z_llm):
        return self.net(z_llm.mean(dim=1)), {}


class FusionPredictor(nn.Module):
    def __init__(self, d_ts=D_TS, d_llm=D_TEXT, d_attn=64, n_heads=4, d_pred=32, dropout=0.1):
        super().__init__()
        self.core = TemporalFusion(d_ts, d_llm, d_attn, n_heads, d_pred, dropout)

    def forward(self, z_ts, z_llm, return_per_head=False):
        return self.core(z_ts, z_llm, return_per_head=return_per_head)


def build_model(name, **kwargs):
    if name == "ts_only":
        return TSOnlyPredictor(D_TS, **kwargs)
    if name == "text_only":
        return TextOnlyPredictor(D_TEXT, **kwargs)
    if name == "fusion":
        return FusionPredictor(**kwargs)
    raise ValueError(f"modelo desconhecido: {name}")
