import sys, pickle, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor

from src.backtest.engine import decompose_alpha_beta
from src.backtest.visualization import BG, PANEL, GOLD, GREEN, RED, WHITE, CYAN, PURPLE, ORANGE, BLUE

OUT_DIR = ROOT / 'src' / 'strategy' / 'trained' / 'images'
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / 'src' / 'strategy' / 'trained' / 'checkpoints'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

embs = np.load(str(ROOT / 'cache/text/top50_daily_embeddings.npy'), mmap_mode='r')
meta = pd.read_csv(ROOT / 'cache/text/top50_daily_metadata.csv')
meta['date'] = pd.to_datetime(meta['date'], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
meta = meta.dropna(subset=['date'])
embs = embs[meta.index]

cached = pd.read_pickle(ROOT / 'cache/data/top50_prepared_data.pkl')
ret_df = cached['ret_df']; known_ret = cached['known_ret']; mask = cached['mask']
split = 2694

emb_cols = [f'emb_{i}' for i in range(embs.shape[1])]
emb_df = pd.DataFrame(embs, columns=emb_cols)
emb_df['ticker'] = meta['ticker'].values; emb_df['date'] = meta['date'].values
daily_emb = emb_df.groupby(['ticker', 'date'], as_index=False)[emb_cols].mean()

tickers_list = sorted(daily_emb['ticker'].unique())
common_dates = sorted(set(daily_emb['date'].unique()) & set(ret_df.index))
train_dates = common_dates[:split]; test_dates = common_dates[split:]

X_tr_list, y_tr_list = [], []
for t in tickers_list:
    emb_t = daily_emb[daily_emb['ticker'] == t].set_index('date')[emb_cols]
    emb_t = emb_t.reindex(common_dates).ffill().bfill()
    if t not in ret_df.columns: continue
    y = ret_df[t].reindex(common_dates)
    X_tr = emb_t.loc[train_dates].values; y_tr = y.loc[train_dates].values
    valid = ~np.isnan(y_tr) & ~np.isnan(X_tr).any(axis=1)
    if valid.sum() > 50:
        X_tr_list.append(X_tr[valid]); y_tr_list.append(y_tr[valid])

X_train = np.vstack(X_tr_list); y_train = np.hstack(y_tr_list)
print(f'Train: {len(y_train)} samples, {X_train.shape[1]} features')

scaler = StandardScaler(); X_s = scaler.fit_transform(X_train)
pca = PCA(n_components=32); X_p = pca.fit_transform(X_s)

models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.001, max_iter=5000),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
    'MLP (64,32)': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42),
}

try:
    from xgboost import XGBRegressor
    models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, verbosity=0, random_state=42)
except ImportError:
    pass

results = []

for name, model in models.items():
    t0 = time.time()
    model.fit(X_p, y_train)
    fit_time = time.time() - t0

    train_pred = model.predict(X_p)
    train_corr = np.corrcoef(train_pred, y_train)[0, 1]

    signal_df = pd.DataFrame(0.0, index=common_dates, columns=tickers_list)
    for t in tickers_list:
        emb_t = daily_emb[daily_emb['ticker'] == t].set_index('date')[emb_cols]
        emb_t = emb_t.reindex(common_dates).ffill().bfill()
        X = np.nan_to_num(emb_t.values, 0.0)
        Xs = scaler.transform(X); Xp = pca.transform(Xs)
        preds = model.predict(Xp)
        signal_df[t] = np.clip(preds, -0.5, 0.5)

    mask_a = mask.reindex(index=common_dates, columns=tickers_list).fillna(0.0)
    ret_a = ret_df.reindex(index=common_dates, columns=tickers_list).fillna(0.0)
    kn_a = known_ret.reindex(index=common_dates, columns=tickers_list).fillna(0.0)

    def tilt_fn(raw):
        x = (raw * mask_a).fillna(0.0)
        x = x.sub(x.mean(axis=1), axis=0)
        x = x.div(x.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        return x

    tilt = tilt_fn(signal_df)
    bh_w = mask_a.div(mask_a.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    bh_ret = (bh_w * ret_a.fillna(0.0)).sum(axis=1)
    mkt_trend = kn_a.mean(axis=1).rolling(20, min_periods=10).mean().fillna(0.0)
    exposure = pd.Series(np.where(mkt_trend > 0, 1.30, 1.00), index=common_dates)
    w = (bh_w + 0.05 * tilt).clip(lower=0.0)
    w = w.div(w.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    base = (w * ret_a.fillna(0.0)).sum(axis=1)
    sr = exposure * base

    split_dt = common_dates[split]
    s_test = sr.loc[split_dt:]; bh_test = bh_ret.loc[split_dt:]

    def stats(r):
        r = r.fillna(0.0); t = (1 + r).prod() - 1
        a = (1 + t) ** (252 / max(len(r), 1)) - 1; v = r.std(ddof=0) * np.sqrt(252)
        return t, a, v, a / v if v > 0 else 0

    s_t, s_a, s_v, s_sh = stats(s_test)
    b_t, b_a, b_v, b_sh = stats(bh_test)
    ab = decompose_alpha_beta(s_test, bh_test, 252)

    results.append({
        'model': name,
        'train_corr': train_corr,
        'fit_time_s': fit_time,
        'test_return_pct': s_t * 100,
        'bh_return_pct': b_t * 100,
        'excess_pct': (s_t - b_t) * 100,
        'sharpe': s_sh,
        'bh_sharpe': b_sh,
        'alpha_pct': ab['alpha_pct'],
        'beta': ab['beta'],
        'r2': ab['r_squared'],
    })

    with open(MODEL_DIR / f'{name.lower().replace(" ","_")}.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'pca': pca}, f)

    print(f'{name:15s} | train_corr={train_corr:.4f} | test={s_t*100:+.1f}% vs BH={b_t*100:+.1f}% | excess={(s_t-b_t)*100:+.1f}% | alpha={ab["alpha_pct"]:+.2f}% | {fit_time:.1f}s')

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / 'model_comparison.csv', index=False)

fig, ax = plt.subplots(figsize=(20, 10))
fig.patch.set_facecolor(BG)
eq_bh_plot = (1 + bh_ret).cumprod() * 100
for i, (_, row) in enumerate(res_df.iterrows()):
    eq = (1 + sr).cumprod() * 100
    ax.plot(eq.index, eq.values, color=[GOLD, CYAN, PURPLE, ORANGE, BLUE, RED][i % 6], lw=2,
            label=f'{row["model"]} ({row["test_return_pct"]:+.1f}%)')
ax.plot(eq_bh_plot.index, eq_bh_plot.values, color=WHITE, lw=1.2, ls='--', alpha=0.5,
        label=f'Buy & Hold ({b_t*100:+.1f}%)')
ax.axvline(split_dt, color=WHITE, ls=':', alpha=0.3)
ax.axhline(100, color='white', ls='--', alpha=0.15)
ax.set_title('All Trained Models vs Buy & Hold — 50 tickers', color='white', fontsize=14, fontweight='bold')
ax.set_facecolor(PANEL); ax.tick_params(colors='white', labelsize=9)
ax.grid(True, alpha=0.12); ax.legend(fontsize=10, facecolor=PANEL, edgecolor='white', labelcolor='white', loc='upper left')
fig.tight_layout(); fig.savefig(OUT_DIR / 'all_models_vs_bh.png', dpi=300, facecolor=BG, edgecolor='none', bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(figsize=(20, 7))
fig.patch.set_facecolor(BG)
excess_vals = res_df['excess_pct'].values
bars = ax.barh(range(len(res_df)), excess_vals, color=[GREEN if v > 0 else RED for v in excess_vals], alpha=0.85, edgecolor='white', lw=0.3)
ax.set_yticks(range(len(res_df))); ax.set_yticklabels(res_df['model'], color='white', fontsize=11)
ax.axvline(0, color='white', lw=0.5)
for i, v in enumerate(excess_vals):
    ax.text(v + (0.3 if v >= 0 else -3), i, f'{v:+.1f}%', va='center', color='white', fontsize=10)
ax.set_title('Excess Return vs Buy & Hold by Model', color='white', fontsize=14, fontweight='bold')
ax.set_facecolor(PANEL); ax.tick_params(colors='white', labelsize=9)
ax.grid(True, alpha=0.12, axis='x')
fig.tight_layout(); fig.savefig(OUT_DIR / 'model_excess_comparison.png', dpi=300, facecolor=BG, edgecolor='none', bbox_inches='tight'); plt.close()

print(f'\nResults: {OUT_DIR}/model_comparison.csv')
print(f'Plots: {OUT_DIR}/')
print(f'Models: {MODEL_DIR}/')
print('\n' + res_df.to_string(index=False))
