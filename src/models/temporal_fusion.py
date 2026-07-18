import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalFusion(nn.Module):
    def __init__(self, d_ts, d_llm, d_attn=64, n_heads=4, d_pred=32, dropout=0.1, out_dim=1):
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
        self.regime_proj = nn.Linear(1, d_attn)
        self.regime_gate = nn.Sequential(
            nn.Linear(d_attn * 3, d_attn),
            nn.ReLU(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid()
        )
        self.pred_head = nn.Sequential(
            nn.Linear(d_attn, d_pred),
            nn.ReLU(),
            nn.Linear(d_pred, out_dim)
        )

    def forward(self, z_ts, z_llm, regime=None, return_attn=False):
        z_ts_proj = self.ts_proj(z_ts)
        z_llm_proj = self.llm_proj(z_llm)
        attn_ts_to_news, w_ts_to_news = self.cross_attn_ts_to_news(
            query=z_ts_proj,
            key=z_llm_proj,
            value=z_llm_proj,
            need_weights=True
        )
        attn_news_to_ts, w_news_to_ts = self.cross_attn_news_to_ts(
            query=z_llm_proj,
            key=z_ts_proj,
            value=z_ts_proj,
            need_weights=True
        )
        g_ts = self.gate_ts(torch.cat([z_ts_proj, attn_ts_to_news], dim=-1))
        g_news = self.gate_news(torch.cat([z_llm_proj, attn_news_to_ts], dim=-1))
        z_ts_fused = z_ts_proj * g_ts + attn_ts_to_news * (1.0 - g_ts)
        z_news_fused = z_llm_proj * g_news + attn_news_to_ts * (1.0 - g_news)
        pooled_ts = torch.mean(z_ts_fused, dim=1)
        pooled_news = torch.mean(z_news_fused, dim=1)
        if regime is not None:
            if regime.dim() == 1:
                regime = regime.unsqueeze(-1)
            r_proj = self.regime_proj(regime)
            r_gate = self.regime_gate(torch.cat([pooled_ts, pooled_news, r_proj], dim=-1))
            z_fused = pooled_ts * (1.0 - r_gate) + pooled_news * r_gate
        else:
            z_fused = pooled_ts + pooled_news
        pred = self.pred_head(z_fused)
        if return_attn:
            return pred, pooled_ts, pooled_news, w_ts_to_news, w_news_to_ts
        return pred, pooled_ts, pooled_news

def compute_alignment_loss(z_ts_proj, z_llm_proj, method="cosine", temperature=0.07):
    if method == "cosine":
        cos_sim = F.cosine_similarity(z_ts_proj, z_llm_proj, dim=-1)
        return 1.0 - torch.mean(cos_sim)
    elif method == "contrastive":
        z_ts_norm = F.normalize(z_ts_proj, p=2, dim=-1)
        z_llm_norm = F.normalize(z_llm_proj, p=2, dim=-1)
        sim_matrix = torch.matmul(z_ts_norm, z_llm_norm.T) / temperature
        labels = torch.arange(z_ts_proj.size(0)).to(z_ts_proj.device)
        loss_ts = F.cross_entropy(sim_matrix, labels)
        loss_llm = F.cross_entropy(sim_matrix.T, labels)
        return 0.5 * (loss_ts + loss_llm)
    else:
        raise ValueError(f"Unknown method {method}")

def compute_total_loss(task_loss, alignment_loss, alpha=0.1):
    return task_loss + alpha * alignment_loss
