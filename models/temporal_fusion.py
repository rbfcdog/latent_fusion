
import torch
import torch.nn as nn


class TemporalFusion(nn.Module):
    def __init__(self, d_ts, d_llm, d_attn=64, n_heads=4, d_pred=32, dropout=0.1):
        super().__init__()

        self.d_attn = d_attn
        self.ts_proj = nn.Linear(d_ts, d_attn)
        self.llm_proj = nn.Linear(d_llm, d_attn)

        self.cross_attn_ts_to_news = nn.MultiheadAttention(
            embed_dim=d_attn,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout
        )

        self.cross_attn_news_to_ts = nn.MultiheadAttention(
            embed_dim=d_attn,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout
        )

        self.gate_ts = nn.Sequential(
            nn.Linear(d_attn * 2, d_attn),
            nn.Sigmoid()
        )

        self.gate_news = nn.Sequential(
            nn.Linear(d_attn * 2, d_attn),
            nn.Sigmoid()
        )

        self.pred_head = nn.Sequential(
            nn.Linear(d_attn, d_pred),
            nn.ReLU(),
            nn.Linear(d_pred, 1)
        )

    def forward(self, z_ts, z_llm, return_per_head=False):
        batch_size = z_ts.shape[0]
        z_ts_proj = self.ts_proj(z_ts)
        z_llm_proj = self.llm_proj(z_llm)

        attn_ts_to_news, attn_w_ts_to_news = self.cross_attn_ts_to_news(
            query=z_ts_proj,
            key=z_llm_proj,
            value=z_llm_proj,
            need_weights=True,
            average_attn_weights=not return_per_head
        )

        attn_news_to_ts, attn_w_news_to_ts = self.cross_attn_news_to_ts(
            query=z_llm_proj,
            key=z_ts_proj,
            value=z_ts_proj,
            need_weights=True,
            average_attn_weights=not return_per_head
        )

        gate_ts = self.gate_ts(torch.cat([z_ts_proj, attn_ts_to_news], dim=-1))
        gate_news = self.gate_news(torch.cat([z_llm_proj, attn_news_to_ts], dim=-1))

        z_ts_fused = z_ts_proj * gate_ts + attn_ts_to_news * (1 - gate_ts)
        z_news_fused = z_llm_proj * gate_news + attn_news_to_ts * (1 - gate_news)

        z_fused = torch.mean(z_ts_fused, dim=1) + torch.mean(z_news_fused, dim=1)

        pred = self.pred_head(z_fused)

        return pred, {
            'attn_ts_to_news': attn_w_ts_to_news,
            'attn_news_to_ts': attn_w_news_to_ts,
            'z_ts_fused': z_ts_fused,
            'z_news_fused': z_news_fused
        }


