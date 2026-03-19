
import torch.nn as nn


class TemporalFusion(nn.Module):
    def __init__(self, d_ts, d_llm, d_attn=64, n_heads=4, d_pred=32):
        super().__init__()

        self.ts_proj = nn.Linear(d_ts, d_attn)
        self.llm_proj = nn.Linear(d_llm, d_attn)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_attn,
            num_heads=n_heads, 
            batch_first=True
        )

        self.pred_head = nn.Sequential(
            nn.Linear(d_attn, d_pred),
            nn.ReLU(),
            nn.Linear(d_pred, 1)
        )

    def forward(self, z_ts, z_llm):
        pass


