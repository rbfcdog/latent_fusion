import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

from src.backtest.engine import BacktestEngine, BacktestConfig, decompose_alpha_beta
from src.strategy import SmaCrossStrategy, MeanReversionStrategy, RegimeRouterStrategy, S1Hard70Strategy
from src.backtest.monte_carlo import monte_carlo_simulate, MonteCarloConfig
from src.backtest.optimizer import grid_search, walk_forward_optimize, WalkForwardConfig
from src.backtest.visualization import BG, PANEL, GOLD, GREEN, RED, WHITE, CYAN, PURPLE, ORANGE, BLUE

OUT_DIR = ROOT / 'src' / 'strategy' / 'trained' / 'images'
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

def generate_synthetic_data(n=2000, seed=42):
    np.random.seed(seed)
    dates = pd.date_range('2018-01-01', periods=n, freq='D')
    mu, sigma = 0.0003, 0.012
    returns = np.random.randn(n) * sigma + mu
    returns[600:750] -= 0.004
    returns[1200:1400] += 0.005
    price = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        'timestamp': dates, 'open': price * (1 + np.random.randn(n) * 0.002),
        'high': price * (1 + np.abs(np.random.randn(n) * 0.003)),
        'low': price * (1 - np.abs(np.random.randn(n) * 0.003)),
        'close': price,
        'volume': np.random.randint(1000, 100000, n),
    })

def get_real_data(ticker='VALE', period='5y'):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    df = tk.history(period=period)
    df.columns = [c.lower() for c in df.columns]
    df['timestamp'] = df.index
    return df.dropna(subset=['close']).sort_values('timestamp')

def time_split_test(df, strategy_class, param_grid, config, n_splits=5):
    n = len(df)
    fold_size = n // (n_splits + 1)
    results = []
    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]
        if len(train_df) < 100 or len(test_df) < 20: continue
        gs = grid_search(train_df, strategy_class, param_grid, config=config, verbose=False)
        best_strat = strategy_class(**gs.best_params)
        r = BacktestEngine(config).run(test_df, best_strat)
        m = r.metrics
        results.append({
            'fold': i + 1,
            'train_days': len(train_df), 'test_days': len(test_df),
            'best_params': gs.best_params,
            'train_sharpe': gs.best_score,
            'test_sharpe': m['sharpe'],
            'test_return_pct': m['total_return_pct'],
            'bh_return_pct': m['bh_return_pct'],
            'excess_pct': m['excess_return_pct'],
            'alpha_pct': m['alpha_pct'], 'beta': m['beta'],
            'max_dd_pct': m['max_drawdown_pct'],
        })
    return pd.DataFrame(results)

def plot_time_split(df_results, model_name, out_path):
    if df_results.empty: return
    fig, ax = plt.subplots(figsize=(20, 7))
    fig.patch.set_facecolor(BG)
    x = np.arange(len(df_results)); w = 0.25
    s_vals = df_results['test_sharpe'].values
    e_vals = df_results['excess_pct'].values
    d_vals = [-abs(v) for v in df_results['max_dd_pct'].values]
    ax.bar(x - w, s_vals, w, color=GOLD, alpha=0.85, label='Sharpe')
    ax.bar(x, e_vals, w, color=CYAN, alpha=0.85, label='Excess %')
    ax.bar(x + w, d_vals, w, color=RED, alpha=0.5, label='Max DD %')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(len(df_results))], color='white', fontsize=9)
    ax.axhline(0, color='white', lw=0.5)
    ax.set_title(f'Time Split Test — {model_name}', color='white', fontsize=14, fontweight='bold')
    ax.set_facecolor(PANEL); ax.tick_params(colors='white', labelsize=9)
    ax.grid(True, alpha=0.12, axis='y')
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor='white', labelcolor='white', loc='upper right')
    fig.tight_layout(); fig.savefig(out_path, dpi=300, facecolor=BG, edgecolor='none', bbox_inches='tight'); plt.close()

def plot_param_heatmap(gs_result, title, out_path):
    pivot = gs_result.all_results.pivot_table(values='_score', index=list(gs_result.best_params.keys())[0],
                                                columns=list(gs_result.best_params.keys())[1] if len(gs_result.best_params) > 1 else list(gs_result.best_params.keys())[0],
                                                aggfunc='mean')
    if isinstance(pivot, pd.Series): return
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG)
    im = ax.imshow(pivot.values, aspect='auto', cmap='magma')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f'{pivot.values[i,j]:.2f}', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, color='white', fontsize=9)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, color='white', fontsize=9)
    ax.set_title(title, color='white', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax); cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    fig.tight_layout(); fig.savefig(out_path, dpi=300, facecolor=BG, edgecolor='none', bbox_inches='tight'); plt.close()

config = BacktestConfig(initial_cash=10000, fee_bps=4, slippage_bps=1, periods_per_year=252)

print('=== 1. SYNTHETIC DATA: ALL STRATEGIES TIME SPLIT ===')
df_syn = generate_synthetic_data(2000)
strategies_params = {
    'SMA Cross': (SmaCrossStrategy, {'fast_window': [10, 20, 50, 100], 'slow_window': [50, 100, 200]}),
    'Mean Reversion': (MeanReversionStrategy, {'lookback': [20, 50, 100], 'z_entry': [1.0, 1.5, 2.0], 'z_exit': [0.3, 0.5]}),
    'Regime Router': (RegimeRouterStrategy, {'vol_window': [10, 20, 40], 'mom_window': [3, 5, 10, 20], 'regime_percentile': [40, 50, 60]}),
    'S1 Hard70': (S1Hard70Strategy, {'vol_window': [10, 20, 30], 'mom_window': [3, 5, 10], 'gate_percentile': [60, 70, 80]}),
}

all_ts_results = []
for name, (cls, grid) in strategies_params.items():
    ts = time_split_test(df_syn, cls, grid, config, n_splits=5)
    ts['model'] = name
    all_ts_results.append(ts)
    plot_time_split(ts, name, OUT_DIR / f'timesplit_{name.lower().replace(" ","_")}.png')
    print(f'{name:20s} | folds={len(ts)} | avg_sharpe={ts["test_sharpe"].mean():.3f} | avg_excess={ts["excess_pct"].mean():+.1f}%')

ts_all = pd.concat(all_ts_results, ignore_index=True)
ts_all.to_csv(OUT_DIR / 'time_split_results.csv', index=False)

print('\n=== 2. PARAMETER HEATMAPS ===')
for name, (cls, grid) in strategies_params.items():
    gs = grid_search(df_syn, cls, grid, config=config, verbose=False)
    plot_param_heatmap(gs, f'{name} — Sharpe Heatmap', OUT_DIR / f'heatmap_{name.lower().replace(" ","_")}.png')
    print(f'{name:20s} | best={gs.best_params} | sharpe={gs.best_score:.3f}')

print('\n=== 3. REAL DATA: VALE 5Y ===')
df_vale = get_real_data('VALE', '5y')
s_v = int(len(df_vale) * 0.7)
df_vale_test = df_vale.iloc[s_v:]
engine = BacktestEngine(config)

for name, (cls, grid) in strategies_params.items():
    if name == 'S1 Hard70': gs = grid_search(df_vale.iloc[:s_v], cls, grid, config=config, verbose=False)
    else: gs = grid_search(df_vale.iloc[:s_v], cls, grid, config=config, verbose=False)
    best = cls(**gs.best_params)
    r = engine.run(df_vale_test, best)
    m = r.metrics
    print(f'{name:20s} | best={gs.best_params} | ret={m["total_return_pct"]:+.1f}% vs BH={m["bh_return_pct"]:+.1f}% | excess={m["excess_return_pct"]:+.1f}% | sharpe={m["sharpe"]:.2f} | alpha={m["alpha_pct"]:+.1f}%')

print(f'\nDone. All outputs in {OUT_DIR}/')
