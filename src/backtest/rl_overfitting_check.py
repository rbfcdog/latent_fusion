import sys, os, json, warnings, time
from pathlib import Path
from copy import deepcopy

_root = Path(os.getcwd())
while not ((_root / 'src').exists() and (_root / 'pyproject.toml').exists()) and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.backtest.visualization import BG, PANEL, GOLD, GREEN, RED, WHITE, CYAN, PURPLE, ORANGE, BLUE
from src.models.rl_env import TradingEnv, REWARD_PRESETS, make_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

pd.set_option('display.max_columns', 80)
pd.set_option('display.width', 160)
_images = _root / 'images'
_images.mkdir(exist_ok=True)

cached = pd.read_pickle(_root / 'cache/data/top50_prepared_data.pkl')
ret_df = cached['ret_df']
mask = cached['mask']
split = 2694

bh_w = mask.div(mask.sum(axis=1).replace(0.0, float('nan')), axis=0).fillna(0.0)
bh_ret = (bh_w * ret_df.fillna(0.0)).sum(axis=1)
price_port = (1 + bh_ret).cumprod() * 100.0

rng = np.random.default_rng(42)
df_port = pd.DataFrame({
    'timestamp': ret_df.index,
    'close': price_port.values,
    'open': price_port.values * (1 + rng.standard_normal(len(price_port)) * 0.001),
    'high': price_port.values * (1 + np.abs(rng.standard_normal(len(price_port))) * 0.002),
    'low': price_port.values * (1 - np.abs(rng.standard_normal(len(price_port))) * 0.002),
    'volume': np.ones(len(price_port)) * 1e6,
})

df_train = df_port.iloc[:split].copy()
df_test = df_port.iloc[split:].copy()

PRESET = "balanced"
TIMESTEPS = 50_000

def evaluate_agent(model, df_eval, preset=PRESET):
    env = make_env(df_eval, preset=preset, lookback=20)
    obs, _ = env.reset()
    done = False
    pv = [env.initial_balance]
    actions = []
    rewards = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))
        pv.append(info['portfolio_value'])
        actions.append(int(action))
        rewards.append(reward)
        done = done or trunc
    pv = np.array(pv)
    daily_rets = np.diff(pv) / pv[:-1]
    total_ret = (pv[-1] / pv[0] - 1) * 100
    sharpe = (np.mean(daily_rets) / (np.std(daily_rets) + 1e-8)) * np.sqrt(252)
    cum = np.cumprod(1 + daily_rets)
    dd = (cum / np.maximum.accumulate(cum) - 1) * 100
    max_dd = float(dd.min())
    n_buy = sum(1 for a in actions if a == 1)
    n_sell = sum(1 for a in actions if a == 2)
    n_hold = sum(1 for a in actions if a == 0)
    return {
        'return_pct': total_ret, 'sharpe': sharpe, 'max_dd_pct': max_dd,
        'n_buy': n_buy, 'n_sell': n_sell, 'n_hold': n_hold,
        'final_portfolio': pv[-1], 'pv': pv, 'actions': actions, 'rewards': rewards,
    }

def train_ppo(df_tr, preset, seed, timesteps=TIMESTEPS):
    env = make_env(df_tr, preset=preset, lookback=20)
    env.reset(seed=seed)
    vec_env = DummyVecEnv([lambda: env])
    model = PPO('MlpPolicy', vec_env, learning_rate=3e-4, verbose=0, device='auto',
                n_steps=2048, batch_size=256, gamma=0.99, ent_coef=0.01, seed=seed)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    return model

def random_agent(df_eval, preset=PRESET, seed=0):
    rng = np.random.default_rng(seed)
    env = make_env(df_eval, preset=preset, lookback=20)
    obs, _ = env.reset(seed=seed)
    done = False
    pv = [env.initial_balance]
    actions = []
    while not done:
        action = int(rng.integers(0, 3))
        obs, reward, done, trunc, info = env.step(action)
        pv.append(info['portfolio_value'])
        actions.append(action)
        done = done or trunc
    pv = np.array(pv)
    daily_rets = np.diff(pv) / pv[:-1]
    total_ret = (pv[-1] / pv[0] - 1) * 100
    sharpe = (np.mean(daily_rets) / (np.std(daily_rets) + 1e-8)) * np.sqrt(252)
    return {'return_pct': total_ret, 'sharpe': sharpe, 'pv': pv, 'actions': actions}

bh_train_ret = (df_train['close'].iloc[-1] / df_train['close'].iloc[0] - 1) * 100
bh_test_ret = (df_test['close'].iloc[-1] / df_test['close'].iloc[0] - 1) * 100
print(f"BH train: {bh_train_ret:+.2f}%  |  BH test: {bh_test_ret:+.2f}%")
print(f"Train: {len(df_train)} days  |  Test: {len(df_test)} days")

print("\n=== TEST 1: Train vs Test Performance Gap ===")
model_t1 = train_ppo(df_train, PRESET, seed=42)
train_eval = evaluate_agent(model_t1, df_train, PRESET)
test_eval = evaluate_agent(model_t1, df_test, PRESET)
gap_return = train_eval['return_pct'] - test_eval['return_pct']
gap_sharpe = train_eval['sharpe'] - test_eval['sharpe']
print(f"Train: return={train_eval['return_pct']:+.2f}%  sharpe={train_eval['sharpe']:.3f}  buys={train_eval['n_buy']}  sells={train_eval['n_sell']}  holds={train_eval['n_hold']}")
print(f"Test:  return={test_eval['return_pct']:+.2f}%  sharpe={test_eval['sharpe']:.3f}  buys={test_eval['n_buy']}  sells={test_eval['n_sell']}  holds={test_eval['n_hold']}")
print(f"Gap:   return={gap_return:+.2f}%  sharpe={gap_sharpe:+.3f}")
print(f"VERDICT: {'OVERFITTING' if abs(gap_return) > 50 or abs(gap_sharpe) > 1.0 else 'NO OVERFITTING'} (gap threshold: 50% return, 1.0 Sharpe)")

print("\n=== TEST 2: Multi-Seed Stability (5 seeds) ===")
seed_results = []
for seed in [42, 123, 456, 789, 2024]:
    t0 = time.time()
    m = train_ppo(df_train, PRESET, seed=seed)
    te = evaluate_agent(m, df_test, PRESET)
    seed_results.append({
        'seed': seed, 'return_pct': te['return_pct'], 'sharpe': te['sharpe'],
        'max_dd_pct': te['max_dd_pct'], 'n_buy': te['n_buy'], 'n_sell': te['n_sell'],
        'n_hold': te['n_hold'], 'time_s': round(time.time() - t0, 0),
    })
    print(f"  seed={seed}: return={te['return_pct']:+.2f}%  sharpe={te['sharpe']:.3f}  dd={te['max_dd_pct']:.1f}%  buys={te['n_buy']}  sells={te['n_sell']}  holds={te['n_hold']}  ({time.time()-t0:.0f}s)")

seeds_df = pd.DataFrame(seed_results)
ret_std = seeds_df['return_pct'].std()
sharpe_std = seeds_df['sharpe'].std()
action_consistency = seeds_df[['n_buy', 'n_sell', 'n_hold']].std().mean()
print(f"\n  Return std: {ret_std:.2f}%  |  Sharpe std: {sharpe_std:.3f}  |  Action std (mean): {action_consistency:.1f}")
print(f"VERDICT: {'UNSTABLE (overfitting risk)' if ret_std > 20 or sharpe_std > 0.5 else 'STABLE'}")

print("\n=== TEST 3: Walk-Forward Validation (5 folds) ===")
n = len(df_port)
fold_size = n // 6
wf_results = []
for i in range(5):
    train_start = 0
    train_end = (i + 1) * fold_size
    test_start = train_end
    test_end = min(test_start + fold_size, n)
    wf_train = df_port.iloc[train_start:train_end].copy()
    wf_test = df_port.iloc[test_start:test_end].copy()
    if len(wf_train) < 200 or len(wf_test) < 50:
        continue
    m = train_ppo(wf_train, PRESET, seed=42, timesteps=30_000)
    te = evaluate_agent(m, wf_test, PRESET)
    bh_fold = (wf_test['close'].iloc[-1] / wf_test['close'].iloc[0] - 1) * 100
    wf_results.append({
        'fold': i + 1, 'train_days': len(wf_train), 'test_days': len(wf_test),
        'return_pct': te['return_pct'], 'bh_return_pct': bh_fold,
        'excess_pct': te['return_pct'] - bh_fold, 'sharpe': te['sharpe'],
        'max_dd_pct': te['max_dd_pct'], 'n_buy': te['n_buy'], 'n_sell': te['n_sell'],
        'n_hold': te['n_hold'],
    })
    print(f"  Fold {i+1}: train={len(wf_train)}d test={len(wf_test)}d  return={te['return_pct']:+.2f}%  BH={bh_fold:+.2f}%  excess={te['return_pct']-bh_fold:+.2f}%  sharpe={te['sharpe']:.3f}  buys={te['n_buy']}  sells={te['n_sell']}  holds={te['n_hold']}")

wf_df = pd.DataFrame(wf_results)
wf_excess_mean = wf_df['excess_pct'].mean()
wf_excess_std = wf_df['excess_pct'].std()
wf_positive_folds = (wf_df['excess_pct'] > -5).sum()
print(f"\n  WF excess: mean={wf_excess_mean:+.2f}%  std={wf_excess_std:.2f}%  folds within -5% of BH: {wf_positive_folds}/{len(wf_df)}")
print(f"VERDICT: {'OVERFITTING (excess varies wildly)' if wf_excess_std > 30 else 'CONSISTENT (no overfitting)'}")

print("\n=== TEST 4: Random Agent Baseline ===")
random_results = []
for seed in range(10):
    r = random_agent(df_test, PRESET, seed=seed)
    random_results.append({'seed': seed, 'return_pct': r['return_pct'], 'sharpe': r['sharpe']})
random_df = pd.DataFrame(random_results)
random_mean_ret = random_df['return_pct'].mean()
random_std_ret = random_df['return_pct'].std()
random_mean_sharpe = random_df['sharpe'].mean()
ppo_test_ret = test_eval['return_pct']
ppo_test_sharpe = test_eval['sharpe']
z_score = (ppo_test_ret - random_mean_ret) / (random_std_ret + 1e-8)
print(f"  Random agent: mean_return={random_mean_ret:+.2f}%  std={random_std_ret:.2f}%  mean_sharpe={random_mean_sharpe:.3f}")
print(f"  PPO agent:    return={ppo_test_ret:+.2f}%  sharpe={ppo_test_sharpe:.3f}")
print(f"  Z-score (PPO vs random): {z_score:.2f}")
print(f"VERDICT: {'PPO IS RANDOM (overfitting)' if abs(z_score) < 1.0 else 'PPO IS SIGNIFICANTLY DIFFERENT FROM RANDOM'}")

print("\n=== TEST 5: Permutation Test (shuffled returns) ===")
df_shuffled = df_train.copy()
shuffled_close = df_shuffled['close'].values.copy()
returns = np.diff(shuffled_close) / shuffled_close[:-1]
rng_perm = np.random.default_rng(999)
rng_perm.shuffle(returns)
shuffled_close[1:] = shuffled_close[0] * np.cumprod(1 + returns)
df_shuffled['close'] = shuffled_close
df_shuffled['open'] = shuffled_close * (1 + rng_perm.standard_normal(len(shuffled_close)) * 0.001)
df_shuffled['high'] = shuffled_close * (1 + np.abs(rng_perm.standard_normal(len(shuffled_close))) * 0.002)
df_shuffled['low'] = shuffled_close * (1 - np.abs(rng_perm.standard_normal(len(shuffled_close))) * 0.002)

model_perm = train_ppo(df_shuffled, PRESET, seed=42)
perm_train = evaluate_agent(model_perm, df_shuffled, PRESET)
perm_test = evaluate_agent(model_perm, df_test, PRESET)
perm_bh_train = (df_shuffled['close'].iloc[-1] / df_shuffled['close'].iloc[0] - 1) * 100
print(f"  Shuffled train BH: {perm_bh_train:+.2f}%")
print(f"  Permuted model on shuffled train: return={perm_train['return_pct']:+.2f}%  sharpe={perm_train['sharpe']:.3f}  buys={perm_train['n_buy']}  sells={perm_train['n_sell']}  holds={perm_train['n_hold']}")
print(f"  Permuted model on real test:      return={perm_test['return_pct']:+.2f}%  sharpe={perm_test['sharpe']:.3f}  buys={perm_test['n_buy']}  sells={perm_test['n_sell']}  holds={perm_test['n_hold']}")
print(f"VERDICT: {'OVERFITTING (model learned shuffled noise)' if perm_train['return_pct'] > 20 and abs(perm_train['return_pct'] - perm_bh_train) > 20 else 'NOT OVERFITTING (model does not learn from noise)'}")

print("\n=== TEST 6: Training Curve Monitoring ===")
class EvalCallback(BaseCallback):
    def __init__(self, df_eval, preset, eval_freq=5000, verbose=0):
        super().__init__(verbose)
        self.df_eval = df_eval
        self.preset = preset
        self.eval_freq = eval_freq
        self.train_rewards = []
        self.eval_returns = []
        self.eval_sharpes = []
        self.eval_steps = []
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            env = make_env(self.df_eval, preset=self.preset, lookback=20)
            obs, _ = env.reset()
            done = False
            pv = [env.initial_balance]
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = env.step(int(action))
                pv.append(info['portfolio_value'])
                done = done or trunc
            pv = np.array(pv)
            daily_rets = np.diff(pv) / pv[:-1]
            total_ret = (pv[-1] / pv[0] - 1) * 100
            sharpe = (np.mean(daily_rets) / (np.std(daily_rets) + 1e-8)) * np.sqrt(252)
            self.eval_returns.append(total_ret)
            self.eval_sharpes.append(sharpe)
            self.eval_steps.append(self.n_calls)
        return True

callback = EvalCallback(df_test, PRESET, eval_freq=5000)
env_train = make_env(df_train, PRESET, lookback=20)
env_train.reset(seed=42)
vec_env = DummyVecEnv([lambda: env_train])
model_curve = PPO('MlpPolicy', vec_env, learning_rate=3e-4, verbose=0, device='auto',
                   n_steps=2048, batch_size=256, gamma=0.99, ent_coef=0.01, seed=42)
model_curve.learn(total_timesteps=TIMESTEPS, callback=callback, progress_bar=False)

curve_steps = callback.eval_steps
curve_returns = callback.eval_returns
curve_sharpes = callback.eval_sharpes
print("  Step  |  Test Return  |  Test Sharpe")
for s, r, sh in zip(curve_steps, curve_returns, curve_sharpes):
    print(f"  {s:6d}  |  {r:+10.2f}%  |  {sh:.3f}")
ret_increase = curve_returns[-1] - curve_returns[0] if len(curve_returns) > 1 else 0
ret_peak = max(curve_returns) if curve_returns else 0
ret_final = curve_returns[-1] if curve_returns else 0
decline_from_peak = ret_peak - ret_final
print(f"\n  Return change: {curve_returns[0]:+.2f}% -> {curve_returns[-1]:+.2f}%  (delta={ret_increase:+.2f}%)")
print(f"  Peak return: {ret_peak:+.2f}%  |  Decline from peak: {decline_from_peak:.2f}%")
print(f"VERDICT: {'OVERFITTING (performance peaked then declined)' if decline_from_peak > 30 else 'STABLE (no degradation)'}")

print("\n" + "=" * 80)
print("=== OVERALL OVERFITTING ASSESSMENT ===")
print("=" * 80)
verdicts = {
    'Train/Test Gap': abs(gap_return) > 50 or abs(gap_sharpe) > 1.0,
    'Seed Stability': ret_std > 20 or sharpe_std > 0.5,
    'Walk-Forward': wf_excess_std > 30,
    'vs Random': abs(z_score) < 1.0,
    'Permutation': perm_train['return_pct'] > 20 and abs(perm_train['return_pct'] - perm_bh_train) > 20,
    'Training Curve': decline_from_peak > 30,
}
n_overfit = sum(verdicts.values())
for test, is_overfit in verdicts.items():
    status = "OVERFITTING" if is_overfit else "PASS"
    print(f"  {test:20s}: {status}")
print(f"\n  Tests indicating overfitting: {n_overfit}/6")
if n_overfit == 0:
    print("  FINAL VERDICT: NO OVERFITTING DETECTED — model generalizes")
elif n_overfit <= 2:
    print("  FINAL VERDICT: MILD OVERFITTING RISK — monitor closely")
else:
    print("  FINAL VERDICT: OVERFITTING DETECTED — model needs regularization")

print("\n=== Generating plots ===")
fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.patch.set_facecolor(BG)

ax = axes[0, 0]
ax.set_facecolor(PANEL)
x = ['Train', 'Test']
y_ret = [train_eval['return_pct'], test_eval['return_pct']]
y_bh = [bh_train_ret, bh_test_ret]
ax.bar([0, 1], y_ret, 0.35, color=[GOLD, CYAN], alpha=0.85, label='PPO')
ax.bar([0.35, 1.35], y_bh, 0.35, color=[WHITE, WHITE], alpha=0.4, label='BH')
ax.set_xticks([0.17, 1.17])
ax.set_xticklabels(x, color='white', fontsize=13)
ax.set_title('Test 1: Train vs Test Gap', color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='white', labelsize=12)
ax.legend(fontsize=11, facecolor=PANEL, edgecolor='white', labelcolor='white')
ax.grid(True, alpha=0.12)
ax.axhline(0, color='white', lw=0.5)

ax = axes[0, 1]
ax.set_facecolor(PANEL)
ax.bar(range(len(seeds_df)), seeds_df['return_pct'], 0.6, color=GOLD, alpha=0.85)
ax.axhline(seeds_df['return_pct'].mean(), color=WHITE, ls='--', lw=1, label=f"Mean={seeds_df['return_pct'].mean():.1f}%")
ax.fill_between(range(len(seeds_df)), 
                seeds_df['return_pct'].mean() - ret_std,
                seeds_df['return_pct'].mean() + ret_std, color=WHITE, alpha=0.1)
ax.set_xticks(range(len(seeds_df)))
ax.set_xticklabels([f's={s}' for s in seeds_df['seed']], color='white', fontsize=11)
ax.set_title(f'Test 2: Multi-Seed (std={ret_std:.1f}%)', color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='white', labelsize=12)
ax.legend(fontsize=11, facecolor=PANEL, edgecolor='white', labelcolor='white')
ax.grid(True, alpha=0.12)
ax.axhline(0, color='white', lw=0.5)

ax = axes[0, 2]
ax.set_facecolor(PANEL)
x_wf = range(len(wf_df))
w = 0.35
ax.bar([i - w/2 for i in x_wf], wf_df['return_pct'], w, color=GOLD, alpha=0.85, label='PPO')
ax.bar([i + w/2 for i in x_wf], wf_df['bh_return_pct'], w, color=WHITE, alpha=0.4, label='BH')
ax.set_xticks(list(x_wf))
ax.set_xticklabels([f'F{i+1}' for i in x_wf], color='white', fontsize=12)
ax.set_title(f'Test 3: Walk-Forward (excess std={wf_excess_std:.1f}%)', color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='white', labelsize=12)
ax.legend(fontsize=11, facecolor=PANEL, edgecolor='white', labelcolor='white')
ax.grid(True, alpha=0.12, axis='y')
ax.axhline(0, color='white', lw=0.5)

ax = axes[1, 0]
ax.set_facecolor(PANEL)
ax.hist(random_df['return_pct'], bins=10, color=WHITE, alpha=0.4, edgecolor='white', label='Random agents')
ax.axvline(ppo_test_ret, color=GOLD, lw=2.5, label=f'PPO={ppo_test_ret:.1f}%')
ax.axvline(random_mean_ret, color=RED, ls='--', lw=1.5, label=f'Random mean={random_mean_ret:.1f}%')
ax.axvline(bh_test_ret, color=GREEN, ls=':', lw=1.5, label=f'BH={bh_test_ret:.1f}%')
ax.set_title(f'Test 4: vs Random (z={z_score:.2f})', color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='white', labelsize=12)
ax.legend(fontsize=10, facecolor=PANEL, edgecolor='white', labelcolor='white')
ax.grid(True, alpha=0.12)

ax = axes[1, 1]
ax.set_facecolor(PANEL)
labels = ['Shuffled\nTrain', 'Real\nTest']
perm_rets = [perm_train['return_pct'], perm_test['return_pct']]
perm_bh = [perm_bh_train, bh_test_ret]
ax.bar([0, 1], perm_rets, 0.35, color=[PURPLE, CYAN], alpha=0.85, label='Permuted PPO')
ax.bar([0.35, 1.35], perm_bh, 0.35, color=[WHITE, WHITE], alpha=0.4, label='BH')
ax.set_xticks([0.17, 1.17])
ax.set_xticklabels(labels, color='white', fontsize=12)
ax.set_title('Test 5: Permutation (noise learning)', color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='white', labelsize=12)
ax.legend(fontsize=10, facecolor=PANEL, edgecolor='white', labelcolor='white')
ax.grid(True, alpha=0.12)
ax.axhline(0, color='white', lw=0.5)

ax = axes[1, 2]
ax.set_facecolor(PANEL)
ax.plot(curve_steps, curve_returns, color=GOLD, lw=2, marker='o', markersize=4)
ax.axhline(bh_test_ret, color=GREEN, ls='--', lw=1, label=f'BH={bh_test_ret:.1f}%')
ax.set_title('Test 6: Training Curve (test return over time)', color='white', fontsize=14, fontweight='bold')
ax.set_xlabel('Timesteps', color='white', fontsize=12)
ax.tick_params(colors='white', labelsize=12)
ax.legend(fontsize=11, facecolor=PANEL, edgecolor='white', labelcolor='white')
ax.grid(True, alpha=0.12)
ax.axhline(0, color='white', lw=0.5)

fig.suptitle('RL Overfitting Validation — 6 Tests (PPO Balanced Preset)', color='white', fontsize=16, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(_images / 'rl_overfitting_validation.png', dpi=300, facecolor=BG, edgecolor='none', bbox_inches='tight')
plt.close()
print(f"Saved: {_images / 'rl_overfitting_validation.png'}")

results_summary = {
    'test1_train_test_gap': {
        'train_return': round(train_eval['return_pct'], 2),
        'test_return': round(test_eval['return_pct'], 2),
        'gap': round(gap_return, 2),
        'verdict': 'OVERFITTING' if verdicts['Train/Test Gap'] else 'PASS',
    },
    'test2_seed_stability': {
        'seeds': seeds_df.to_dict('records'),
        'return_std': round(ret_std, 2),
        'sharpe_std': round(sharpe_std, 3),
        'verdict': 'UNSTABLE' if verdicts['Seed Stability'] else 'STABLE',
    },
    'test3_walk_forward': {
        'folds': wf_df.to_dict('records'),
        'excess_mean': round(wf_excess_mean, 2),
        'excess_std': round(wf_excess_std, 2),
        'verdict': 'OVERFITTING' if verdicts['Walk-Forward'] else 'CONSISTENT',
    },
    'test4_random_baseline': {
        'random_mean_return': round(random_mean_ret, 2),
        'ppo_return': round(ppo_test_ret, 2),
        'z_score': round(z_score, 2),
        'verdict': 'RANDOM' if verdicts['vs Random'] else 'SIGNIFICANT',
    },
    'test5_permutation': {
        'shuffled_train_return': round(perm_train['return_pct'], 2),
        'shuffled_bh_return': round(perm_bh_train, 2),
        'real_test_return': round(perm_test['return_pct'], 2),
        'verdict': 'OVERFITTING' if verdicts['Permutation'] else 'PASS',
    },
    'test6_training_curve': {
        'initial_test_return': round(curve_returns[0], 2) if curve_returns else 0,
        'peak_test_return': round(ret_peak, 2),
        'final_test_return': round(ret_final, 2),
        'decline_from_peak': round(decline_from_peak, 2),
        'verdict': 'OVERFITTING' if verdicts['Training Curve'] else 'STABLE',
    },
    'n_overfit_indicators': n_overfit,
    'final_verdict': 'NO OVERFITTING' if n_overfit == 0 else ('MILD RISK' if n_overfit <= 2 else 'OVERFITTING'),
}

with open(_images / 'rl_overfitting_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)
print(f"Saved: {_images / 'rl_overfitting_results.json'}")
print("\n=== DONE ===")
