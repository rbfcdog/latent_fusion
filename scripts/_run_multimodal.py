import sys, os, json, time
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer
from src.backtest.engine import decompose_alpha_beta
from src.backtest.visualization import BG, PANEL, GOLD, GREEN, RED, WHITE, CYAN

news_dir = _root / 'data/text'
price_dir = _root / 'data/time_series'
text_tickers = {f.stem.upper() for f in news_dir.glob('*.jsonl')}
price_tickers = {f.stem.upper() for f in price_dir.glob('*.csv')}
tickers = sorted(text_tickers & price_tickers)[:6]
print(f'Tickers: {len(tickers)}')

all_articles = []
for t in tickers:
    fp = news_dir / f'{t}.jsonl'
    with open(fp) as f:
        for line in f:
            try:
                doc = json.loads(line.strip())
                d = doc.get('Date',''); txt = doc.get('Article','') or doc.get('Article_title','')
                if d and txt and len(txt)>30: all_articles.append({'ticker':t,'date':str(d)[:10],'text':txt[:800]})
            except: continue
df = pd.DataFrame(all_articles)
df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
df = df.dropna(subset=['date'])
daily_text = df.groupby(['ticker','date'])['text'].apply(lambda x: ' '.join(x)[:3000]).reset_index()
daily_text['_idx'] = range(len(daily_text))

t0 = time.time()
st_model = SentenceTransformer('all-MiniLM-L6-v2')
text_embs = st_model.encode(daily_text['text'].tolist(), show_progress_bar=False)
print(f'Text embeddings: {text_embs.shape} ({time.time()-t0:.1f}s)')

dates_all = sorted(daily_text['date'].unique())
pframes = []
for t in tickers:
    ts_file = price_dir / f'{t.lower()}.csv'
    px = pd.read_csv(ts_file, usecols=['Date','Close'])
    px['date'] = pd.to_datetime(px['Date'], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
    px = px.dropna(subset=['date','Close']).sort_values('date'); px['ticker'] = t
    pframes.append(px[['date','ticker','Close']])
prices = pd.concat(pframes, ignore_index=True)
px_panel = prices.pivot_table(index='date', columns='ticker', values='Close', aggfunc='last')

cd = sorted(set(dates_all) & set(px_panel.index))
n_dates = len(cd); n_tickers = len(tickers)
print(f'Common dates: {n_dates} | {cd[0].date()} -> {cd[-1].date()}')

emb_dim = text_embs.shape[1]
emb_3d = np.zeros((n_dates, n_tickers, emb_dim))
for j, t in enumerate(tickers):
    sub = daily_text[daily_text['ticker'] == t]
    emb_map = {}
    for _, row in sub.iterrows():
        d = row['date']; idx = int(row['_idx'])
        if d in cd and idx < len(text_embs):
            if d in emb_map: emb_map[d].append(text_embs[idx])
            else: emb_map[d] = [text_embs[idx]]
    last = np.zeros(emb_dim)
    for i, d in enumerate(cd):
        if d in emb_map: last = np.mean(emb_map[d], axis=0)
        emb_3d[i,j,:] = last

px = px_panel.loc[cd]
ret_raw = px.pct_change().fillna(0.0)
known_ret = ret_raw.shift(1).fillna(0.0); ret_df = ret_raw.shift(-1).fillna(0.0)
mask = (~px.isna()).astype(float)
mkt_ret = known_ret.mean(axis=1)
vol_20 = mkt_ret.rolling(20).std()
regime = (vol_20 > vol_20.median()).astype(float).fillna(0.0)
reg_mat = pd.DataFrame(np.repeat(regime.values[:,None], n_tickers, axis=1), index=cd, columns=tickers)
rev1 = -np.sign(known_ret).fillna(0.0); mom3 = np.sign(known_ret.rolling(3, min_periods=2).mean()).fillna(0.0)
mom10 = np.sign(known_ret.rolling(10, min_periods=5).mean()).fillna(0.0)

# TEXT STRATEGY
drift = np.zeros((n_dates, n_tickers))
for i in range(1, n_dates): drift[i] = np.linalg.norm(emb_3d[i]-emb_3d[i-1], axis=1)
drift_df = pd.DataFrame(drift, index=cd, columns=tickers)
p95 = float(np.percentile(drift_df.values.flatten(), 95))
events = (drift_df >= p95).astype(float)
intensity = pd.DataFrame(0.0, index=cd, columns=tickers)
for j in range(n_tickers):
    ev = events.iloc[:,j].values; lam = 0.0
    for i in range(n_dates): lam = np.exp(-0.2)*lam + 0.8*ev[i]; intensity.iloc[i,j] = lam
int_rank = intensity.rank(axis=1, pct=True); hard70 = (int_rank >= 0.70).astype(float)
raw_text = hard70 * (reg_mat * rev1 + (1.0 - reg_mat) * mom3)

# NUMERICAL STRATEGY (multi-feature drift as proxy for MOMENT+OpenAI fusion)
feat1 = known_ret.rolling(5).std().fillna(0.0)
feat2 = known_ret.rolling(10).mean().fillna(0.0)
feat3 = known_ret.rolling(20).std().fillna(0.0)
feat4 = known_ret.fillna(0.0)
num_feat = np.stack([feat1.values, feat2.values, feat3.values, feat4.values], axis=-1)
num_feat = np.nan_to_num(num_feat, 0.0)
num_drift = np.zeros((n_dates, n_tickers))
for i in range(1, n_dates): num_drift[i] = np.linalg.norm(num_feat[i]-num_feat[i-1], axis=1)
num_drift_df = pd.DataFrame(num_drift, index=cd, columns=tickers)
p95_num = float(np.percentile(num_drift_df.values.flatten(), 95))
num_events = (num_drift_df >= p95_num).astype(float)
num_intensity = pd.DataFrame(0.0, index=cd, columns=tickers)
for j in range(n_tickers):
    ev = num_events.iloc[:,j].values; lam = 0.0
    for i in range(n_dates): lam = np.exp(-0.2)*lam + 0.8*ev[i]; num_intensity.iloc[i,j] = lam
num_int_rank = num_intensity.rank(axis=1, pct=True); num_hard70 = (num_int_rank >= 0.70).astype(float)
raw_numerical = num_hard70 * (reg_mat * rev1 + (1.0 - reg_mat) * mom10)

# PORTFOLIO
def run_pf(raw_signal):
    def tilt_fn(ra):
        x = (ra*mask).fillna(0.0); x = x.sub(x.mean(axis=1), axis=0)
        x = x.div(x.abs().sum(axis=1).replace(0.0,np.nan), axis=0).fillna(0.0)
        return x
    tilt = tilt_fn(raw_signal)
    bh_w = mask.div(mask.sum(axis=1).replace(0.0,np.nan), axis=0).fillna(0.0)
    bh_ret = (bh_w*ret_df.fillna(0.0)).sum(axis=1)
    mkt_trend = known_ret.mean(axis=1).rolling(20,min_periods=10).mean().fillna(0.0)
    exposure = pd.Series(np.where(mkt_trend>0,1.30,1.00), index=cd)
    w = (bh_w+0.05*tilt).clip(lower=0.0)
    w = w.div(w.sum(axis=1).replace(0.0,np.nan), axis=0).fillna(0.0)
    base = (w*ret_df.fillna(0.0)).sum(axis=1)
    return exposure*base, bh_ret

sr_text, bh_ret = run_pf(raw_text)
sr_num, _ = run_pf(raw_numerical)
split_idx = int(n_dates*0.7)

def stats(r):
    r=r.fillna(0.0); t=(1+r).prod()-1
    a=(1+t)**(252/max(len(r),1))-1; v=r.std(ddof=0)*np.sqrt(252)
    return t,a,v,a/v if v>0 else 0

for name, sr in [('Text S1 Hard70', sr_text), ('Numerical (4-feat)', sr_num)]:
    st = sr.iloc[split_idx:]; bht = bh_ret.iloc[split_idx:]
    s_t,s_a,s_v,s_sh = stats(st); b_t,b_a,b_v,b_sh = stats(bht)
    ab = decompose_alpha_beta(st, bht, 252)
    print('{:20s} | ret={:+6.1f}% vs BH={:+6.1f}% | sharpe={:.3f} vs {:.3f} | alpha={:+.2f}% beta={:.3f} R2={:.3f}'.format(
        name, s_t*100, b_t*100, s_sh, b_sh, ab['alpha_pct'], ab['beta'], ab['r_squared']))

images = _root/'images'; images.mkdir(exist_ok=True)
eq_text = (1+sr_text).cumprod()*100; eq_num = (1+sr_num).cumprod()*100; eq_bh = (1+bh_ret).cumprod()*100

fig,ax=plt.subplots(figsize=(20,10))
fig.patch.set_facecolor(BG)
ax.plot(eq_bh.index, eq_bh.values, color=WHITE, lw=1.2, ls='--', alpha=0.5, label='Buy & Hold')
ax.plot(eq_text.index, eq_text.values, color=GOLD, lw=2.5, label='Text S1 Hard70 (SentenceTransformer)')
ax.plot(eq_num.index, eq_num.values, color=CYAN, lw=2.5, label='Numerical (vol+mom+ret drift)')
ax.axvline(cd[split_idx], color=WHITE, ls=':', alpha=0.3)
ax.fill_between(eq_bh.index, eq_bh.values, 100, where=eq_bh.values>=100, color=GREEN, alpha=0.04)
ax.axhline(100, color='white', ls='--', alpha=0.15)
ax.set_title('Text vs Numerical Embedding Strategies — {} tickers {} days'.format(n_tickers,n_dates), color='white', fontsize=14, fontweight='bold')
ax.set_facecolor(PANEL); ax.tick_params(colors='white', labelsize=9)
ax.grid(True, alpha=0.12); ax.legend(fontsize=11, facecolor=PANEL, edgecolor='white', labelcolor='white', loc='upper left')
fig.tight_layout(); fig.savefig(images/'multimodal_strategies_vs_bh.png', dpi=300, facecolor=BG, edgecolor='none', bbox_inches='tight'); plt.close()
print('Plot: images/multimodal_strategies_vs_bh.png')
