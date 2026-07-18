#!/usr/bin/env python3
"""
Script de comparação completo:
  Transformer price-only vs Transformer text+price vs PPO price-only vs PPO text+price vs BH
  Portfolio selecionado: 10 cripto + ADBE + COST
  Embeddings nativos: all-MiniLM-L6-v2 (384-dim -> PCA 10 dims)
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings('ignore')
_root = Path('/home/rodrigodog/latent_fusion')
sys.path.insert(0, str(_root))
from src.models.rl_env import TradingEnv
from sentence_transformers import SentenceTransformer

# ─── 1. DADOS ──────────────────────────────────────────────────────────────
news_dir = _root / 'data/news_crypto'
news_frames = []
for f in sorted(news_dir.glob('*.csv')):
    nd = pd.read_csv(f)
    nd['date'] = pd.to_datetime(nd['date'], errors='coerce').dt.normalize()
    news_frames.append(nd)
news = pd.concat(news_frames, ignore_index=True).dropna(subset=['date','title'])

prices = pd.read_parquet(_root / 'data/lse_market_data/combined_1d.parquet')
prices['timestamp'] = pd.to_datetime(prices['timestamp']).dt.tz_localize(None).dt.normalize()
price_pivot = prices.pivot_table(index='timestamp', columns='symbol', values='close')

# ─── 2. EMBEDDINGS NATIVOS CRIPTO ──────────────────────────────────────────
cache_dir = _root / 'cache/text_crypto'
cache_dir.mkdir(parents=True, exist_ok=True)
if (cache_dir / 'crypto_daily_embeddings.npy').exists():
    crypto_emb = np.load(cache_dir / 'crypto_daily_embeddings.npy')
    crypto_meta = pd.read_csv(cache_dir / 'crypto_daily_metadata.csv')
    crypto_meta['date'] = pd.to_datetime(crypto_meta['date'])
else:
    model_st = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    daily_news = news.groupby(['ticker', 'date'])['title'].apply(' '.join).reset_index()
    crypto_emb = model_st.encode(daily_news['title'].tolist(), batch_size=32, show_progress_bar=False)
    crypto_meta = daily_news[['ticker', 'date']].copy()
    np.save(cache_dir / 'crypto_daily_embeddings.npy', crypto_emb)
    crypto_meta.to_csv(cache_dir / 'crypto_daily_metadata.csv', index=False)

# ─── 3. EMBEDDINGS S&P500 CACHED ───────────────────────────────────────────
meta_sp = pd.read_csv(_root / 'cache/text/top50_daily_metadata.csv')
meta_sp['date'] = pd.to_datetime(meta_sp['date'], errors='coerce').dt.tz_localize(None).dt.normalize()
meta_sp = meta_sp.dropna(subset=['date'])
emb_sp_all = np.load(_root / 'cache/text/top50_daily_embeddings.npy')
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

# ─── 4. PORTFOLIO E ALINHAMENTO ────────────────────────────────────────────
crypto_tickers = sorted(prices[prices['asset_group'] == 'crypto']['symbol'].unique())
sp_covered = [t for t in sp_ov_tickers if
              any(r['ticker'] == t for r in text_rows)]
portfolio = crypto_tickers + sp_covered
portfolio = [t for t in portfolio if t in price_pivot.columns]

port_prices = price_pivot[portfolio].ffill().bfill()
returns = port_prices.pct_change().fillna(0)
returns.index = pd.to_datetime(returns.index).normalize()

# PCA 10 dims on all embeddings
all_embs = np.vstack([r['emb'] for r in text_rows])
pca = PCA(n_components=10, random_state=42)
all_embs_pca = pca.fit_transform(all_embs)
pca_ratio = pca.explained_variance_ratio_.sum()

text_df = pd.DataFrame({'date': [r['date'] for r in text_rows],
                        'ticker': [r['ticker'] for r in text_rows]})
text_df['date'] = pd.to_datetime(text_df['date']).dt.normalize()
emb_cols = [f'te_{i}' for i in range(10)]
text_df[emb_cols] = all_embs_pca

text_pivot = text_df.pivot_table(index='date', columns='ticker', values=emb_cols)
text_pivot.columns = [f'{t}_{c}' for c, t in text_pivot.columns]
text_pivot = text_pivot.reindex(returns.index).fillna(0.0)
port_text_cols = [c for c in text_pivot.columns if c.rsplit('_te_',1)[0] in portfolio]
text_pivot_f = text_pivot[port_text_cols]
coverage = (text_pivot_f.abs().sum(axis=1) > 0).astype(float)

# Normalize
ret_std = returns.std() + 1e-8
returns_norm = ((returns - returns.mean()) / ret_std).clip(-5, 5)
text_std = text_pivot_f.std() + 1e-8
text_norm = ((text_pivot_f - text_pivot_f.mean()) / text_std).clip(-5, 5)

n_assets = len(portfolio)
n_text_feat = text_pivot_f.shape[1]
combined = pd.concat([returns_norm, text_norm, coverage.rename('coverage')], axis=1)
src_dim_combined = combined.shape[1]

# ─── 5. SEQUENCES ──────────────────────────────────────────────────────────
seq_len = 15
X_price_np = returns_norm.values.astype(np.float32)
X_combined_np = combined.values.astype(np.float32)
Y_ret_np = returns_norm.values.astype(np.float32)
raw_ret_np = returns.values.astype(np.float32)

split_idx = int(len(X_price_np) * 0.7)

def make_seqs(X_src, Y_tgt):
    src_list, tgt_list, raw_list = [], [], []
    for i in range(len(X_src) - seq_len - 1):
        src_list.append(X_src[i:i+seq_len])
        tgt_list.append(Y_tgt[i:i+seq_len+1])
        raw_list.append(raw_ret_np[i+seq_len])
    return (torch.tensor(np.array(src_list), dtype=torch.float32),
            torch.tensor(np.array(tgt_list), dtype=torch.float32),
            np.array(raw_list))

src_p, tgt_all, raw_all = make_seqs(X_price_np, Y_ret_np)
src_c, _, _ = make_seqs(X_combined_np, Y_ret_np)

n_seq = len(src_p)
sp = int(n_seq * 0.7)
src_p_tr, src_p_te = src_p[:sp], src_p[sp:]
src_c_tr, src_c_te = src_c[:sp], src_c[sp:]
tgt_tr, tgt_te = tgt_all[:sp], tgt_all[sp:]
raw_tr, raw_te = raw_all[:sp], raw_all[sp:]

print(f"Portfolio ({len(portfolio)}): {portfolio}")
print(f"Sequences: {n_seq} (train {sp} / test {n_seq-sp})")
print(f"PCA variance retained: {pca_ratio:.2%}")
print(f"src_price: {src_p_tr.shape}, src_combined: {src_c_tr.shape}")

# ─── 6. MODELOS TRANSFORMER ────────────────────────────────────────────────
class FusionTransformer(nn.Module):
    def __init__(self, src_dim, tgt_dim, d_model=64, n_heads=4, ffn=256,
                 dropout=0.15, n_enc=2, n_dec=2):
        super().__init__()
        self.src_emb = nn.Sequential(nn.Linear(src_dim, d_model), nn.LayerNorm(d_model))
        self.tgt_emb = nn.Sequential(nn.Linear(tgt_dim, d_model), nn.LayerNorm(d_model))
        self.src_pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.04)
        self.tgt_pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.04)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, ffn, dropout, batch_first=True)
        dec = nn.TransformerDecoderLayer(d_model, n_heads, ffn, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_enc)
        self.decoder = nn.TransformerDecoder(dec, n_dec)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, tgt_dim))

    def forward(self, src, tgt):
        L_s, L_t = src.size(1), tgt.size(1)
        se = self.src_emb(src) + self.src_pos[:, :L_s]
        te = self.tgt_emb(tgt) + self.tgt_pos[:, :L_t]
        mask = nn.Transformer.generate_square_subsequent_mask(L_t).to(src.device)
        mem = self.encoder(se)
        out = self.decoder(te, mem, tgt_mask=mask)
        return self.fc(out)

    def predict_next(self, src):
        self.eval()
        with torch.no_grad():
            B, _, D = src.shape
            tgt = torch.zeros(B, 1, self.tgt_emb[0].in_features, device=src.device)
            out = self.forward(src, tgt)
            return out[:, 0, :].cpu().numpy()

def train_tf(model, src_tr, tgt_tr, src_te, tgt_te, epochs=30, lr=1e-3, bs=32):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.SmoothL1Loss()
    tr_h, te_h = [], []
    dl_tr = DataLoader(TensorDataset(src_tr, tgt_tr), bs, shuffle=True)
    dl_te = DataLoader(TensorDataset(src_te, tgt_te), bs)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0
        for s, t in dl_tr:
            opt.zero_grad()
            crit(model(s, t[:, :-1]), t[:, 1:]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += crit(model(s, t[:, :-1]).detach(), t[:, 1:]).item()
        model.eval()
        te_tot = 0
        with torch.no_grad():
            for s, t in dl_te:
                te_tot += crit(model(s, t[:, :-1]), t[:, 1:]).item()
        tr_h.append(tot/len(dl_tr)); te_h.append(te_tot/len(dl_te))
        sched.step()
    return tr_h, te_h, time.time()-t0

def eval_tf(model, src, raw_ret):
    preds = model.predict_next(src)
    exp_p = np.exp(np.clip(preds * 5, -8, 8))
    w = exp_p / exp_p.sum(axis=1, keepdims=True)
    pr = (w * raw_ret).sum(axis=1)
    bh = raw_ret.mean(axis=1)
    cum = np.cumprod(1 + pr)
    bh_cum = np.cumprod(1 + bh)
    sharpe = np.mean(pr) / (np.std(pr) + 1e-8) * np.sqrt(252)
    dd = (cum / np.maximum.accumulate(cum) - 1).min() * 100
    return {'ret': (cum[-1]-1)*100, 'bh_ret': (bh_cum[-1]-1)*100,
            'excess': (cum[-1]-bh_cum[-1])*100, 'sharpe': sharpe,
            'dd': dd, 'cum': cum, 'bh_cum': bh_cum}

# Train price-only
print("\n[Transformer price-only]")
tf_price = FusionTransformer(src_dim=n_assets, tgt_dim=n_assets)
tr_hp, te_hp, t_price = train_tf(tf_price, src_p_tr, tgt_tr, src_p_te, tgt_te)
ev_p_tr = eval_tf(tf_price, src_p_tr, raw_tr)
ev_p_te = eval_tf(tf_price, src_p_te, raw_te)
print(f"  Train MSE {tr_hp[-1]:.3f} | Test MSE {te_hp[-1]:.3f}")
print(f"  Train ret {ev_p_tr['ret']:+.2f}% | Test ret {ev_p_te['ret']:+.2f}% | Excess {ev_p_te['excess']:+.2f}% | Sharpe {ev_p_te['sharpe']:.3f}")

# Train text+price
print("\n[Transformer text+price]")
tf_fus = FusionTransformer(src_dim=src_dim_combined, tgt_dim=n_assets)
tr_hf, te_hf, t_fus = train_tf(tf_fus, src_c_tr, tgt_tr, src_c_te, tgt_te)
ev_f_tr = eval_tf(tf_fus, src_c_tr, raw_tr)
ev_f_te = eval_tf(tf_fus, src_c_te, raw_te)
print(f"  Train MSE {tr_hf[-1]:.3f} | Test MSE {te_hf[-1]:.3f}")
print(f"  Train ret {ev_f_tr['ret']:+.2f}% | Test ret {ev_f_te['ret']:+.2f}% | Excess {ev_f_te['excess']:+.2f}% | Sharpe {ev_f_te['sharpe']:.3f}")

# ─── 7. PPO ────────────────────────────────────────────────────────────────
def synth_price(raw_rets):
    port = raw_rets.mean(axis=1)
    return pd.DataFrame({'close': np.cumprod(1+port)*100})

def make_text_arr(returns_slice, text_feats_np, start_i, length):
    """Text features sliced to match the RL env day count"""
    end_i = start_i + length
    end_i = min(end_i, len(text_feats_np))
    arr = text_feats_np[start_i:end_i]
    pad = np.zeros((length - len(arr), arr.shape[1]))
    return np.vstack([arr, pad]).astype(np.float32) if len(arr) < length else arr.astype(np.float32)

# Text array for env: mean of all tickers per day (keep it simple for RL obs)
text_feats_daily = text_norm.values.astype(np.float32)  # (days, n_text_feat)
rl_split = int(len(raw_ret_np) * 0.7)

df_tr_rl = synth_price(raw_ret_np[:rl_split])
df_te_rl = synth_price(raw_ret_np[rl_split:])
txt_tr = text_feats_daily[:rl_split]
txt_te = text_feats_daily[rl_split:]

def eval_ppo(agent, df, txt=None):
    env = TradingEnv(df, text_features=txt, lookback=20)
    obs, _ = env.reset()
    done, pv, acts = False, [env.initial_balance], []
    while not done:
        a, _ = agent.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(int(a))
        pv.append(info['portfolio_value'])
        acts.append(int(a))
        done = done or trunc
    pv = np.array(pv)
    dr = np.diff(pv)/pv[:-1]
    cum = np.cumprod(1+dr)
    sh = np.mean(dr)/(np.std(dr)+1e-8)*np.sqrt(252)
    dd = (cum/np.maximum.accumulate(cum)-1).min()*100
    return {'ret': (cum[-1]-1)*100, 'sharpe': sh, 'dd': dd, 'cum': cum,
            'n_buy': sum(1 for a in acts if a==1), 'n_sell': sum(1 for a in acts if a==2)}

print("\n[PPO price-only]")
t0 = time.time()
env_ppo_p = TradingEnv(df_tr_rl, lookback=20)
vec_p = DummyVecEnv([lambda: TradingEnv(df_tr_rl, lookback=20)])
ppo_price = PPO('MlpPolicy', vec_p, learning_rate=3e-4, n_steps=2048,
                batch_size=256, gamma=0.99, ent_coef=0.01, seed=42, verbose=0)
ppo_price.learn(80_000, progress_bar=False)
t_ppo_p = time.time()-t0
ppo_p_tr = eval_ppo(ppo_price, df_tr_rl)
ppo_p_te = eval_ppo(ppo_price, df_te_rl)
print(f"  Train ret {ppo_p_tr['ret']:+.2f}% | Test ret {ppo_p_te['ret']:+.2f}% | Sharpe {ppo_p_te['sharpe']:.3f} | buys/sells {ppo_p_te['n_buy']}/{ppo_p_te['n_sell']}")

print("\n[PPO text+price]")
t0 = time.time()
vec_t = DummyVecEnv([lambda: TradingEnv(df_tr_rl, text_features=txt_tr, lookback=20)])
ppo_text = PPO('MlpPolicy', vec_t, learning_rate=3e-4, n_steps=2048,
               batch_size=256, gamma=0.99, ent_coef=0.01, seed=42, verbose=0)
ppo_text.learn(80_000, progress_bar=False)
t_ppo_t = time.time()-t0
ppo_t_tr = eval_ppo(ppo_text, df_tr_rl, txt_tr)
ppo_t_te = eval_ppo(ppo_text, df_te_rl, txt_te)
print(f"  Train ret {ppo_t_tr['ret']:+.2f}% | Test ret {ppo_t_te['ret']:+.2f}% | Sharpe {ppo_t_te['sharpe']:.3f} | buys/sells {ppo_t_te['n_buy']}/{ppo_t_te['n_sell']}")

# ─── 8. RESULTADOS E GRÁFICO ───────────────────────────────────────────────
bh_te_ret = ev_p_te['bh_ret']
results = {
    'Transformer\nPrice-only':  {'test_ret': ev_p_te['ret'],  'train_ret': ev_p_tr['ret'],  'sharpe': ev_p_te['sharpe'], 'dd': ev_p_te['dd'],  'excess': ev_p_te['excess'],  'time': t_price, 'ovfit': ev_p_tr['ret']-ev_p_te['ret']},
    'Transformer\nText+Price':  {'test_ret': ev_f_te['ret'],  'train_ret': ev_f_tr['ret'],  'sharpe': ev_f_te['sharpe'], 'dd': ev_f_te['dd'],  'excess': ev_f_te['excess'],  'time': t_fus,   'ovfit': ev_f_tr['ret']-ev_f_te['ret']},
    'PPO\nPrice-only':          {'test_ret': ppo_p_te['ret'], 'train_ret': ppo_p_tr['ret'], 'sharpe': ppo_p_te['sharpe'],'dd': ppo_p_te['dd'], 'excess': ppo_p_te['ret']-bh_te_ret, 'time': t_ppo_p, 'ovfit': ppo_p_tr['ret']-ppo_p_te['ret']},
    'PPO\nText+Price':          {'test_ret': ppo_t_te['ret'], 'train_ret': ppo_t_tr['ret'], 'sharpe': ppo_t_te['sharpe'],'dd': ppo_t_te['dd'], 'excess': ppo_t_te['ret']-bh_te_ret, 'time': t_ppo_t, 'ovfit': ppo_t_tr['ret']-ppo_t_te['ret']},
    'Buy & Hold':               {'test_ret': bh_te_ret,       'train_ret': ev_p_tr['bh_ret'], 'sharpe': -0.295,           'dd': -19.7,          'excess': 0.0,                'time': 0,       'ovfit': 0.0},
}

print("\n" + "="*80)
print(f"{'Model':<24} {'Test Ret':>9} {'Train Ret':>10} {'Excess vs BH':>13} {'Sharpe':>8} {'Max DD':>8} {'Overfitting':>12}")
print("-"*80)
for name, r in results.items():
    nm = name.replace('\n', ' ')
    print(f"{nm:<24} {r['test_ret']:>+9.2f}% {r['train_ret']:>+9.2f}% {r['excess']:>+12.2f}% {r['sharpe']:>8.3f} {r['dd']:>7.1f}% {r['ovfit']:>+11.2f}pp")
print("="*80)

# ─── 9. GRÁFICO ────────────────────────────────────────────────────────────
DARK = '#0d0d1a'; SUB = '#1a1a2e'
GOLD='#D4A843'; GREEN='#00E676'; RED='#FF1744'; CYAN='#18FFFF'
PURPLE='#CE93D8'; ORANGE='#FF9800'; WHITE='#AAAAAA'; BLUE='#42A5F5'
PALETTE = [GOLD, CYAN, ORANGE, PURPLE, WHITE]
MODEL_NAMES = list(results.keys())

# Fig 1: Métricas comparativas
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=DARK)
for ax in axes.flat: ax.set_facecolor(SUB)
fig.suptitle('Comparação: Transformer vs RL (PPO) + Embedding Textual\n'
             f'Portfólio: 10 Cripto + ADBE + COST  |  PCA reteve {pca_ratio:.0%} da variância  |  {len(portfolio)} ativos',
             color=WHITE, fontsize=14, y=1.01)

labels = [n.replace('\n', '\n') for n in MODEL_NAMES]
colors = PALETTE

def bar_ax(ax, vals, title, ylabel, fmt='{:.1f}', ref=None, ref_label=None):
    bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.85)
    if ref is not None:
        ax.axhline(ref, color=RED, ls='--', lw=1.5, label=ref_label or '')
        ax.legend(fontsize=11, facecolor=DARK, labelcolor=WHITE)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, color=WHITE, fontsize=11)
    ax.set_title(title, color=WHITE, fontsize=14, pad=8)
    ax.set_ylabel(ylabel, color=WHITE, fontsize=12)
    ax.tick_params(colors=WHITE, labelsize=11)
    ax.grid(axis='y', alpha=0.12)
    for spine in ax.spines.values(): spine.set_edgecolor('#333355')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                fmt.format(val)+'%', ha='center', va='bottom', color=WHITE, fontsize=11)

test_rets = [r['test_ret'] for r in results.values()]
excess = [r['excess'] for r in results.values()]
sharpes = [r['sharpe'] for r in results.values()]
dds = [r['dd'] for r in results.values()]

bar_ax(axes[0,0], test_rets, 'Retorno no Teste (%)', 'Retorno (%)', fmt='{:+.1f}', ref=0, ref_label='Zero')
bar_ax(axes[0,1], excess, 'Excess Return vs Buy & Hold (%)', 'Excess (%)', fmt='{:+.1f}', ref=0, ref_label='BH baseline')
bar_ax(axes[1,0], sharpes, 'Sharpe Ratio (Test)', 'Sharpe', fmt='{:.2f}', ref=0)
bar_ax(axes[1,1], dds, 'Max Drawdown (%)', 'Drawdown (%)', fmt='{:.1f}')

plt.tight_layout()
plt.savefig(_root / 'images/text_embedding_comparison_metrics.png',
            dpi=300, facecolor=DARK, edgecolor='none', bbox_inches='tight')
plt.close()
print("Saved: images/text_embedding_comparison_metrics.png")

# Fig 2: Equity curves (test period)
n_te = len(ev_p_te['cum'])
x_ax = np.arange(n_te)
ppo_p_padded = np.pad(ppo_p_te['cum'], (0, max(0, n_te-len(ppo_p_te['cum']))), constant_values=ppo_p_te['cum'][-1])[:n_te]
ppo_t_padded = np.pad(ppo_t_te['cum'], (0, max(0, n_te-len(ppo_t_te['cum']))), constant_values=ppo_t_te['cum'][-1])[:n_te]

fig2, ax2 = plt.subplots(figsize=(14, 7), facecolor=DARK)
ax2.set_facecolor(SUB)
ax2.plot(x_ax, ev_p_te['bh_cum'][:n_te], color=WHITE, lw=1.5, ls='--', label='Buy & Hold', alpha=0.6)
ax2.plot(x_ax, ev_p_te['cum'],            color=GOLD,  lw=2.0, label='Transformer Price-only')
ax2.plot(x_ax, ev_f_te['cum'],            color=CYAN,  lw=2.5, label='Transformer Text+Price', zorder=5)
ax2.plot(x_ax, ppo_p_padded,             color=ORANGE,lw=2.0, label='PPO Price-only')
ax2.plot(x_ax, ppo_t_padded,             color=PURPLE,lw=2.0, label='PPO Text+Price')
ax2.axhline(1.0, color=WHITE, ls=':', lw=1, alpha=0.4)
ax2.fill_between(x_ax, ev_f_te['cum'], 1.0,
                 where=ev_f_te['cum'] >= 1.0, alpha=0.08, color=GREEN)
ax2.fill_between(x_ax, ev_f_te['cum'], 1.0,
                 where=ev_f_te['cum'] < 1.0, alpha=0.08, color=RED)
ax2.set_title('Curvas de Equity — Período de Teste (out-of-sample)\n'
              'Embedding textual nativo: all-MiniLM-L6-v2 → PCA 10 dims  |  12 ativos (10 cripto + ADBE + COST)',
              color=WHITE, fontsize=14)
ax2.set_xlabel('Dias de Teste', color=WHITE, fontsize=12)
ax2.set_ylabel('Crescimento do Capital (base=1)', color=WHITE, fontsize=12)
ax2.tick_params(colors=WHITE, labelsize=12)
ax2.legend(fontsize=14, facecolor=DARK, labelcolor=WHITE)
ax2.grid(alpha=0.12)
for spine in ax2.spines.values(): spine.set_edgecolor('#333355')
plt.tight_layout()
plt.savefig(_root / 'images/text_embedding_equity_curves.png',
            dpi=300, facecolor=DARK, edgecolor='none', bbox_inches='tight')
plt.close()
print("Saved: images/text_embedding_equity_curves.png")

# Fig 3: Heatmap — ganho textual por modelo
fig3, ax3 = plt.subplots(figsize=(10, 6), facecolor=DARK)
ax3.set_facecolor(SUB)
tf_gain = ev_f_te['ret'] - ev_p_te['ret']
ppo_gain = ppo_t_te['ret'] - ppo_p_te['ret']
tf_sh_gain = ev_f_te['sharpe'] - ev_p_te['sharpe']
ppo_sh_gain = ppo_t_te['sharpe'] - ppo_p_te['sharpe']

categories = ['Retorno (%)', 'Sharpe', 'Max DD (pp)']
tf_gains  = [tf_gain,  tf_sh_gain,  ev_f_te['dd'] - ev_p_te['dd']]
ppo_gains = [ppo_gain, ppo_sh_gain, ppo_t_te['dd'] - ppo_p_te['dd']]

x_pos = np.arange(len(categories))
w = 0.35
b1 = ax3.bar(x_pos - w/2, tf_gains,  w, color=CYAN,   alpha=0.85, label='Transformer')
b2 = ax3.bar(x_pos + w/2, ppo_gains, w, color=ORANGE, alpha=0.85, label='PPO')
ax3.axhline(0, color=WHITE, ls='--', lw=1.2, alpha=0.5)
ax3.set_xticks(x_pos); ax3.set_xticklabels(categories, color=WHITE, fontsize=14)
ax3.set_title('Ganho do Embedding Textual\n(com texto − sem texto, no teste)', color=WHITE, fontsize=16)
ax3.set_ylabel('Ganho Absoluto', color=WHITE, fontsize=14)
ax3.tick_params(colors=WHITE, labelsize=12)
ax3.legend(fontsize=14, facecolor=DARK, labelcolor=WHITE)
ax3.grid(axis='y', alpha=0.12)
for bar in list(b1)+list(b2):
    v = bar.get_height()
    ax3.text(bar.get_x()+bar.get_width()/2, v + (0.02 if v>=0 else -0.08),
             f'{v:+.2f}', ha='center', va='bottom', color=WHITE, fontsize=11)
for spine in ax3.spines.values(): spine.set_edgecolor('#333355')
plt.tight_layout()
plt.savefig(_root / 'images/text_embedding_gain.png',
            dpi=300, facecolor=DARK, edgecolor='none', bbox_inches='tight')
plt.close()
print("Saved: images/text_embedding_gain.png")

# ─── 10. JSON RESULTADOS ───────────────────────────────────────────────────
import json
out = {
    'portfolio': portfolio, 'n_assets': len(portfolio), 'pca_variance': round(pca_ratio, 4),
    'text_feat_dim': n_text_feat, 'seq_len': seq_len,
    'train_days': rl_split, 'test_days': len(raw_ret_np)-rl_split,
    'models': {k.replace('\n',' '): {m: round(v,4) for m,v in r.items() if not isinstance(v, np.ndarray)}
               for k, r in results.items()},
    'text_gain': {
        'transformer_return_pp': round(tf_gain, 4),
        'transformer_sharpe': round(tf_sh_gain, 4),
        'ppo_return_pp': round(ppo_gain, 4),
        'ppo_sharpe': round(ppo_sh_gain, 4),
    }
}
with open(_root / 'cache/text_model_comparison.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Saved: cache/text_model_comparison.json")
