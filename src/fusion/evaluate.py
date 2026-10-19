import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def predict(model, ds, device, batch_size=512, return_attn=False):
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size)
    preds, ent_ts, ent_txt = [], [], []
    with torch.no_grad():
        for z_ts, z_llm, _ in dl:
            z_ts, z_llm = z_ts.to(device), z_llm.to(device)
            pred, attn = model(z_ts, z_llm)
            preds.append(pred.squeeze(1).cpu().numpy())
            if return_attn and attn:
                for key, out in (("attn_ts_to_news", ent_ts), ("attn_news_to_ts", ent_txt)):
                    w = attn[key]
                    w = w / (w.sum(dim=-1, keepdim=True) + 1e-10)
                    ent = -(w * torch.log(w + 1e-10)).sum(dim=-1).mean(dim=-1)
                    out.append(ent.cpu().numpy())
    preds = np.concatenate(preds)
    if return_attn and ent_ts:
        return preds, np.concatenate(ent_ts), np.concatenate(ent_txt)
    return preds, None, None


def regression_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    err = p - y
    out = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "sign_acc": float(np.mean(np.sign(p) == np.sign(y))),
        "ic": float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 1e-12 else 0.0,
    }
    ybin = (y > 0).astype(int)
    if 0 < ybin.mean() < 1 and np.std(p) > 1e-12:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(ybin, p))
    else:
        out["auc"] = np.nan
    return out


def backtest_metrics(df):
    daily = df.groupby("date").agg(
        strat=("ret_strat", "mean"), bh=("r_fwd", "mean")
    ).sort_index()
    peak = daily["strat"].cumsum()
    bh_peak = daily["bh"].cumsum()

    def _sharpe(x):
        return float(x.mean() / (x.std() + 1e-12) * np.sqrt(252))

    def _mdd(x):
        eq = (1 + x).cumprod()
        return float(((eq / eq.cummax()) - 1).min())

    beta = float(np.cov(daily["strat"], daily["bh"])[0, 1] / (daily["bh"].var() + 1e-12))
    alpha = float(252 * (daily["strat"].mean() - beta * daily["bh"].mean()))
    return {
        "total_ret_pct": float(((1 + daily["strat"]).prod() - 1) * 100),
        "bh_ret_pct": float(((1 + daily["bh"]).prod() - 1) * 100),
        "excess_ret_pct": float((((1 + daily["strat"]).prod() - (1 + daily["bh"]).prod())) * 100),
        "sharpe": _sharpe(daily["strat"]),
        "bh_sharpe": _sharpe(daily["bh"]),
        "max_dd_pct": _mdd(daily["strat"]) * 100,
        "win_rate": float((daily["strat"] > 0).mean()),
        "alpha_ann": alpha,
        "beta": beta,
    }, daily


def per_regime_metrics(df):
    rows = {}
    for reg, sub in df.groupby("regime"):
        m = regression_metrics(sub["r_fwd"].values, sub["pred"].values)
        m["n"] = len(sub)
        rows[int(reg)] = m
    return rows


def causal_text_effect(model, ds, device, seed=42, batch_size=512):
    torch.manual_seed(seed)
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size)
    deltas, preds_clean, preds_shuf, ys = [], [], [], []
    with torch.no_grad():
        for z_ts, z_llm, y in dl:
            z_ts, z_llm = z_ts.to(device), z_llm.to(device)
            p_clean, _ = model(z_ts, z_llm)
            perm = torch.randperm(z_llm.shape[0], device=device)
            p_shuf, _ = model(z_ts, z_llm[perm])
            deltas.append((p_clean - p_shuf).abs().squeeze(1).cpu().numpy())
            preds_clean.append(p_clean.squeeze(1).cpu().numpy())
            preds_shuf.append(p_shuf.squeeze(1).cpu().numpy())
            ys.append(y.numpy())
    y = np.concatenate(ys)
    ic_clean = np.corrcoef(np.concatenate(preds_clean), y)[0, 1]
    ic_shuf = np.corrcoef(np.concatenate(preds_shuf), y)[0, 1]
    return {
        "mean_abs_delta_pred": float(np.concatenate(deltas).mean()),
        "ic_clean": float(ic_clean),
        "ic_text_shuffled": float(ic_shuf),
        "delta_ic": float(ic_clean - ic_shuf),
    }


def lead_lag_analysis(panel, max_lag=5):
    rows = []
    for a in panel["arrays"]:
        drift = np.linalg.norm(np.diff(a["text"], axis=0), axis=1)
        drift = np.concatenate([[np.nan], drift])
        s_drift = pd.Series(drift, index=pd.DatetimeIndex(a["dates"]))
        s_ret = pd.Series(a["r_fwd"], index=pd.DatetimeIndex(a["dates"]))
        df = pd.DataFrame({"drift": s_drift, "r_fwd": s_ret}).dropna()
        if len(df) < 100:
            continue
        for lag in range(-max_lag, max_lag + 1):
            corr = df["drift"].corr(df["r_fwd"].shift(-lag))
            rows.append({"lag": lag, "corr": corr})
    out = pd.DataFrame(rows).groupby("lag")["corr"].agg(["mean", "std", "count"])
    return out.reset_index()
