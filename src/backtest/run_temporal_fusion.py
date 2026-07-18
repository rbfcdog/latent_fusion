import sys
import os
import time
import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
_root = Path('/home/rodrigodog/latent_fusion')
sys.path.insert(0, str(_root))

from src.models.temporal_fusion import TemporalFusion, compute_alignment_loss, compute_total_loss

BG = '#0d0d1a'
PANEL = '#1a1a2e'
GOLD = '#D4A843'
GREEN = '#00E676'
RED = '#FF1744'
CYAN = '#18FFFF'
WHITE = '#AAAAAA'
PURPLE = '#CE93D8'
ORANGE = '#FF9800'
BLUE = '#42A5F5'

def main():
    if not (_root / 'data/news_crypto').exists():
        news_dir = _root / 'sample_data/news_crypto'
        prices_path = _root / 'sample_data/prices/combined_1d.parquet'
        cache_dir = _root / 'sample_data/cache/text_crypto'
        meta_sp_path = _root / 'sample_data/cache/text/top50_daily_metadata.csv'
        emb_sp_path = _root / 'sample_data/cache/text/top50_daily_embeddings.npy'
    else:
        news_dir = _root / 'data/news_crypto'
        prices_path = _root / 'data/lse_market_data/combined_1d.parquet'
        cache_dir = _root / 'cache/text_crypto'
        meta_sp_path = _root / 'cache/text/top50_daily_metadata.csv'
        emb_sp_path = _root / 'cache/text/top50_daily_embeddings.npy'

    news_frames = []
    for f in sorted(news_dir.glob('*.csv')):
        nd = pd.read_csv(f)
        nd['date'] = pd.to_datetime(nd['date'], errors='coerce').dt.normalize()
        news_frames.append(nd)
    news = pd.concat(news_frames, ignore_index=True).dropna(subset=['date','title'])

    prices = pd.read_parquet(prices_path)
    prices['timestamp'] = pd.to_datetime(prices['timestamp']).dt.tz_localize(None).dt.normalize()
    price_pivot = prices.pivot_table(index='timestamp', columns='symbol', values='close')

    crypto_emb = np.load(cache_dir / 'crypto_daily_embeddings.npy')
    crypto_meta = pd.read_csv(cache_dir / 'crypto_daily_metadata.csv')
    crypto_meta['date'] = pd.to_datetime(crypto_meta['date'])

    meta_sp = pd.read_csv(meta_sp_path)
    meta_sp['date'] = pd.to_datetime(meta_sp['date'], errors='coerce').dt.tz_localize(None).dt.normalize()
    meta_sp = meta_sp.dropna(subset=['date'])
    emb_sp_all = np.load(emb_sp_path)
    nasdaq_tickers = prices[prices['asset_group'] == 'nasdaq']['symbol'].unique()
    sp_ov_tickers = sorted(set(nasdaq_tickers) & set(meta_sp['ticker'].unique()))
    sp_ov = meta_sp[(meta_sp['date'] >= '2025-01-01') & (meta_sp['ticker'].isin(sp_ov_tickers))]

    text_rows = []
    news_to_price = {t: t.replace('-', '/') for t in news['ticker'].unique()}
    for i, row in crypto_meta.iterrows():
        pt = news_to_price.get(row['ticker'], row['ticker'])
        text_rows.append({'date': row['date'], 'ticker': pt, 'emb': crypto_emb[i]})
    for _, row in sp_ov.iterrows():
        idx = row.name
        text_rows.append({'date': row['date'], 'ticker': row['ticker'], 'emb': emb_sp_all[idx]})

    crypto_tickers = sorted(prices[prices['asset_group'] == 'crypto']['symbol'].unique())
    sp_covered = [t for t in sp_ov_tickers if any(r['ticker'] == t for r in text_rows)]
    portfolio = crypto_tickers + sp_covered
    portfolio = [t for t in portfolio if t in price_pivot.columns]

    port_prices = price_pivot[portfolio].ffill().bfill()
    returns = port_prices.pct_change().fillna(0)
    returns.index = pd.to_datetime(returns.index).normalize()

    all_embs = np.vstack([r['emb'] for r in text_rows])
    pca = PCA(n_components=10, random_state=42)
    all_embs_pca = pca.fit_transform(all_embs)

    text_df = pd.DataFrame({'date': [r['date'] for r in text_rows], 'ticker': [r['ticker'] for r in text_rows]})
    text_df['date'] = pd.to_datetime(text_df['date']).dt.normalize()
    emb_cols = [f'te_{i}' for i in range(10)]
    text_df[emb_cols] = all_embs_pca

    text_pivot = text_df.pivot_table(index='date', columns='ticker', values=emb_cols)
    text_pivot.columns = [f'{t}_{c}' for c, t in text_pivot.columns]
    text_pivot = text_pivot.reindex(returns.index).fillna(0.0)
    port_text_cols = [c for c in text_pivot.columns if c.rsplit('_te_',1)[0] in portfolio]
    text_pivot_f = text_pivot[port_text_cols]

    ret_std = returns.std() + 1e-8
    returns_norm = ((returns - returns.mean()) / ret_std).clip(-5, 5)
    text_std = text_pivot_f.std() + 1e-8
    text_norm = ((text_pivot_f - text_pivot_f.mean()) / text_std).clip(-5, 5)

    market_ret = returns.mean(axis=1)
    market_vol = market_ret.rolling(20).std().fillna(0.0)
    regime_raw = (market_vol > market_vol.rolling(100).median()).astype(float).values

    n_assets = len(portfolio)
    n_text_feat = text_pivot_f.shape[1]

    seq_len = 15
    src_ts_list = []
    src_llm_list = []
    src_reg_list = []
    tgt_list = []
    raw_list = []

    for i in range(len(returns) - seq_len):
        src_ts_list.append(returns_norm.values[i:i+seq_len])
        src_llm_list.append(text_norm.values[i:i+seq_len])
        src_reg_list.append([regime_raw[i+seq_len-1]])
        tgt_list.append(returns_norm.values[i+seq_len])
        raw_list.append(returns.values[i+seq_len])

    ts_tensor = torch.tensor(np.array(src_ts_list), dtype=torch.float32)
    llm_tensor = torch.tensor(np.array(src_llm_list), dtype=torch.float32)
    reg_tensor = torch.tensor(np.array(src_reg_list), dtype=torch.float32)
    tgt_tensor = torch.tensor(np.array(tgt_list), dtype=torch.float32)
    raw_returns = np.array(raw_list)

    n_samples = len(ts_tensor)
    split_idx = int(n_samples * 0.7)

    train_ds = TensorDataset(ts_tensor[:split_idx], llm_tensor[:split_idx], reg_tensor[:split_idx], tgt_tensor[:split_idx])
    test_ds = TensorDataset(ts_tensor[split_idx:], llm_tensor[split_idx:], reg_tensor[split_idx:], tgt_tensor[split_idx:])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 40

    configs = [
        {"name": "Task-Only", "alpha": 0.0, "method": "cosine"},
        {"name": "Cosine-Aligned", "alpha": 0.15, "method": "cosine"},
        {"name": "Contrastive-Aligned", "alpha": 0.15, "method": "contrastive"}
    ]

    results = {}

    for cfg in configs:
        model = TemporalFusion(d_ts=n_assets, d_llm=text_norm.shape[1], d_attn=64, n_heads=4, d_pred=32, out_dim=n_assets)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.SmoothL1Loss()

        train_losses = []
        task_losses = []
        align_losses = []

        for epoch in range(epochs):
            model.train()
            tot_loss = 0.0
            tot_task = 0.0
            tot_align = 0.0
            for z_ts, z_llm, r, target in train_loader:
                z_ts, z_llm, r, target = z_ts.to(device), z_llm.to(device), r.to(device), target.to(device)
                optimizer.zero_grad()
                pred, pts, pllm = model(z_ts, z_llm, r)
                task_loss = criterion(pred, target)
                align_loss = compute_alignment_loss(pts, pllm, method=cfg["method"])
                loss = compute_total_loss(task_loss, align_loss, alpha=cfg["alpha"])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tot_loss += loss.item()
                tot_task += task_loss.item()
                tot_align += align_loss.item()
            train_losses.append(tot_loss / len(train_loader))
            task_losses.append(tot_task / len(train_loader))
            align_losses.append(tot_align / len(train_loader))
            scheduler.step()

        model.eval()
        preds_list = []
        cos_sims = []
        with torch.no_grad():
            for z_ts, z_llm, r, target in test_loader:
                z_ts, z_llm, r = z_ts.to(device), z_llm.to(device), r.to(device)
                pred, pts, pllm = model(z_ts, z_llm, r)
                preds_list.append(pred.cpu().numpy())
                cos_sim = F.cosine_similarity(pts, pllm, dim=-1)
                cos_sims.extend(cos_sim.cpu().numpy())

        preds_all = np.vstack(preds_list)
        cos_sims = np.array(cos_sims)

        raw_test = raw_returns[split_idx:]
        exp_p = np.exp(np.clip(preds_all * 5, -8, 8))
        weights = exp_p / exp_p.sum(axis=1, keepdims=True)
        port_ret = (weights * raw_test).sum(axis=1)
        cum = np.cumprod(1 + port_ret)
        sharpe = np.mean(port_ret) / (np.std(port_ret) + 1e-8) * np.sqrt(252)
        dd = (cum / np.maximum.accumulate(cum) - 1).min() * 100

        results[cfg["name"]] = {
            "test_return_pct": float((cum[-1] - 1) * 100),
            "sharpe": float(sharpe),
            "max_dd_pct": float(dd),
            "cum_curve": cum.tolist(),
            "train_losses": train_losses,
            "task_losses": task_losses,
            "align_losses": align_losses,
            "mean_alignment_score": float(np.mean(cos_sims)),
            "alignment_scores": cos_sims.tolist()
        }

    raw_test = raw_returns[split_idx:]
    bh_ret = raw_test.mean(axis=1)
    bh_cum = np.cumprod(1 + bh_ret)
    bh_sharpe = np.mean(bh_ret) / (np.std(bh_ret) + 1e-8) * np.sqrt(252)
    bh_dd = (bh_cum / np.maximum.accumulate(bh_cum) - 1).min() * 100

    results["Buy & Hold"] = {
        "test_return_pct": float((bh_cum[-1] - 1) * 100),
        "sharpe": float(bh_sharpe),
        "max_dd_pct": float(bh_dd),
        "cum_curve": bh_cum.tolist()
    }

    images_dir = _root / 'images'
    images_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(12, 7))
    plt.gca().set_facecolor(PANEL)
    plt.gcf().patch.set_facecolor(BG)
    for name in ["Task-Only", "Cosine-Aligned", "Contrastive-Aligned"]:
        plt.plot(results[name]["train_losses"], label=f"{name} Total Loss", lw=2)
    plt.title("Training Loss Curves", color='white', fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", color='white', fontsize=14)
    plt.ylabel("Loss", color='white', fontsize=14)
    plt.tick_params(colors='white', labelsize=12)
    plt.grid(True, alpha=0.12)
    plt.legend(facecolor=PANEL, edgecolor='white', labelcolor='white', fontsize=12)
    plt.savefig(images_dir / 'tf_loss_curves.png', dpi=300, facecolor=BG, edgecolor='none', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 7))
    plt.gca().set_facecolor(PANEL)
    plt.gcf().patch.set_facecolor(BG)
    plt.plot(bh_cum, color=WHITE, lw=1.5, ls='--', alpha=0.7, label='Buy & Hold')
    plt.plot(results["Task-Only"]["cum_curve"], color=RED, lw=2, label='Task-Only')
    plt.plot(results["Cosine-Aligned"]["cum_curve"], color=GOLD, lw=2, label='Cosine-Aligned')
    plt.plot(results["Contrastive-Aligned"]["cum_curve"], color=CYAN, lw=2, label='Contrastive-Aligned')
    plt.axhline(1, color='white', ls='--', alpha=0.15)
    plt.title("Test Set Cumulative Returns", color='white', fontsize=16, fontweight='bold')
    plt.xlabel("Trading Day", color='white', fontsize=14)
    plt.ylabel("Equity Value", color='white', fontsize=14)
    plt.tick_params(colors='white', labelsize=12)
    plt.grid(True, alpha=0.12)
    plt.legend(facecolor=PANEL, edgecolor='white', labelcolor='white', fontsize=12)
    plt.savefig(images_dir / 'tf_equity_curves.png', dpi=300, facecolor=BG, edgecolor='none', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 7))
    plt.gca().set_facecolor(PANEL)
    plt.gcf().patch.set_facecolor(BG)
    for name in ["Task-Only", "Cosine-Aligned", "Contrastive-Aligned"]:
        scores = results[name]["alignment_scores"]
        rolling_scores = pd.Series(scores).rolling(20, min_periods=1).mean()
        plt.plot(rolling_scores, label=f"{name} (Mean: {results[name]['mean_alignment_score']:.3f})", lw=2)
    plt.title("Rolling Alignment Score (Cosine Similarity)", color='white', fontsize=16, fontweight='bold')
    plt.xlabel("Trading Day", color='white', fontsize=14)
    plt.ylabel("Similarity", color='white', fontsize=14)
    plt.tick_params(colors='white', labelsize=12)
    plt.grid(True, alpha=0.12)
    plt.legend(facecolor=PANEL, edgecolor='white', labelcolor='white', fontsize=12)
    plt.savefig(images_dir / 'tf_alignment_scores.png', dpi=300, facecolor=BG, edgecolor='none', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 7))
    plt.gca().set_facecolor(PANEL)
    plt.gcf().patch.set_facecolor(BG)
    models = ["Buy & Hold", "Task-Only", "Cosine-Aligned", "Contrastive-Aligned"]
    sharpes = [results[m]["sharpe"] for m in models]
    colors = [WHITE, RED, GOLD, CYAN]
    plt.bar(models, sharpes, color=colors, alpha=0.85)
    plt.axhline(0, color='white', lw=0.5)
    plt.title("Test Sharpe Ratio Comparison", color='white', fontsize=16, fontweight='bold')
    plt.ylabel("Sharpe Ratio", color='white', fontsize=14)
    plt.tick_params(colors='white', labelsize=12)
    plt.grid(True, alpha=0.12, axis='y')
    plt.savefig(images_dir / 'tf_metric_comparison.png', dpi=300, facecolor=BG, edgecolor='none', bbox_inches='tight')
    plt.close()

    with open(_root / 'cache/tf_alignment_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
