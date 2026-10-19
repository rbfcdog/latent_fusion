import numpy as np
import torch
from torch.utils.data import DataLoader


def predict(model, ds, device, batch_size=512, return_attn=False):
    model.eval()
    preds, entropies = [], []
    with torch.no_grad():
        for z_ts, z_text, text_mask, past_vol, _ in DataLoader(ds, batch_size=batch_size):
            z_ts, z_text = z_ts.to(device), z_text.to(device)
            text_mask, past_vol = text_mask.to(device), past_vol.to(device)
            log_pred, attn = model(z_ts, z_text, text_mask, past_vol, return_attn=return_attn)
            preds.append(log_pred.exp().squeeze(1).cpu().numpy())
            if return_attn:
                weights = attn["attn_ts_to_news"][:, 0, :-1] * text_mask
                p = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-10)
                entropy = -(p * p.clamp_min(1e-10).log()).sum(dim=1)
                entropy = entropy.masked_fill(~text_mask.any(dim=1), float("nan"))
                entropies.append(entropy.cpu().numpy())
    return np.concatenate(preds), np.concatenate(entropies) if entropies else None


def volatility_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    mse = np.mean((y - p) ** 2)
    denom = np.sum((y - y.mean()) ** 2)
    return {"n": len(y), "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y - p))),
            "r2": float(1 - np.sum((y - p) ** 2) / denom) if denom > 0 else float("nan")}


def grouped_metrics(df, column):
    return {str(group): volatility_metrics(part["target_vol"], part["pred"])
            for group, part in df.groupby(column)}


def shuffled_text_effect(model, ds, device, seed=42, batch_size=512):
    rng = torch.Generator().manual_seed(seed)
    model.eval()
    clean, shuffled, targets = [], [], []
    with torch.no_grad():
        for z_ts, z_text, mask, past_vol, log_target in DataLoader(ds, batch_size=batch_size):
            z_ts, z_text = z_ts.to(device), z_text.to(device)
            mask, past_vol = mask.to(device), past_vol.to(device)
            perm = torch.randperm(len(z_ts), generator=rng).to(device)
            p_clean, _ = model(z_ts, z_text, mask, past_vol)
            p_shuffled, _ = model(z_ts, z_text[perm], mask[perm], past_vol)
            clean.append(p_clean.exp().squeeze(1).cpu().numpy())
            shuffled.append(p_shuffled.exp().squeeze(1).cpu().numpy())
            targets.append(log_target.exp().numpy())
    y, p, p_shuf = np.concatenate(targets), np.concatenate(clean), np.concatenate(shuffled)
    return {"mean_abs_prediction_change": float(np.mean(np.abs(p - p_shuf))),
            "rmse_clean": volatility_metrics(y, p)["rmse"],
            "rmse_shuffled": volatility_metrics(y, p_shuf)["rmse"]}
