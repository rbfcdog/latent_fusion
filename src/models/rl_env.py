from __future__ import annotations

import numpy as np
import pandas as pd
from gymnasium import Env, spaces


class TradingEnv(Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000,
        lookback: int = 20,
        fee_pct: float = 0.001,
        reward_alpha: float = 0.5,
        reward_beta: float = 2.0,
        reward_gamma: float = 1.0,
        reward_scaling: float = 1e-4,
        reward_type: str = "vol_aware",
        tech_indicator_list: list[str] | None = None,
        text_features: np.ndarray | None = None,
    ):
        self.initial_balance = initial_balance
        self.lookback = lookback
        self.fee_pct = fee_pct
        self.alpha = reward_alpha
        self.beta = reward_beta
        self.gamma = reward_gamma
        self.reward_scaling = reward_scaling
        self.reward_type = reward_type
        self.text_features = text_features
        self.n_text = text_features.shape[1] if text_features is not None else 0
        self.tech_indicator_list = tech_indicator_list or [
            "signal_norm",
            "vol_norm",
            "kl_norm",
            "price_trend",
        ]
        self.df = self._create_features(df).reset_index(drop=True)
        self.n_steps = len(self.df)
        self.action_space = spaces.Discrete(3)
        obs_dim = len(self.tech_indicator_list) + 3 + self.n_text
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.reset()

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["return"] = df["close"].pct_change()
        rolling_mean = df["return"].rolling(self.lookback).mean().fillna(0.0)
        rolling_std = df["return"].rolling(self.lookback).std().fillna(0.0)
        df["signal"] = rolling_mean
        df["vol"] = rolling_std.clip(0.0, 0.05)
        df["kl"] = np.abs(df["return"] - rolling_mean).rolling(3).mean().fillna(0.0)
        df["signal_norm"] = np.tanh(df["signal"] * 10.0)
        df["vol_norm"] = df["vol"] / (df["vol"].max() + 1e-6)
        df["kl_norm"] = np.tanh(df["kl"] * 20.0)
        close_rolling_mean = df["close"].rolling(self.lookback).mean()
        close_rolling_std = df["close"].rolling(self.lookback).std() + 1e-6
        df["price_trend"] = np.tanh(
            (df["close"] - close_rolling_mean) / close_rolling_std
        ).fillna(0.0)
        return df

    def _predicted_vol(self, idx: int) -> float:
        vol_norm = float(self.df.loc[idx, "vol_norm"])
        signal_norm = float(self.df.loc[idx, "signal_norm"])
        return float(np.clip(vol_norm * 0.6 + np.abs(signal_norm) * 0.4, 0.0, 1.0))

    def _get_obs(self) -> np.ndarray:
        if self.day >= self.n_steps:
            self.day = self.n_steps - 1
        tech_obs = []
        for col in self.tech_indicator_list:
            if col in self.df.columns:
                tech_obs.append(float(self.df.loc[self.day, col]))
            else:
                tech_obs.append(0.0)
        close_price = float(self.df.loc[self.day, "close"])
        portfolio_value = self.cash + self.position * close_price
        cash_norm = np.clip(
            (self.cash - self.initial_balance) / (self.initial_balance + 1e-6),
            -5.0,
            5.0,
        )
        position_norm = np.clip(
            (self.position * close_price) / (self.initial_balance + 1e-6),
            -5.0,
            5.0,
        )
        predicted_vol = self._predicted_vol(self.day)
        obs = np.array(
            tech_obs + [float(cash_norm), float(position_norm), predicted_vol],
            dtype=np.float32,
        )
        if self.text_features is not None and self.day < len(self.text_features):
            obs = np.concatenate([obs, self.text_features[self.day].astype(np.float32)])
        return obs

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        self.day = 0
        self.cash = self.initial_balance
        self.position = 0.0
        self.asset_memory = [self.initial_balance]
        self.position_memory = [0.0]
        self.cash_memory = [self.initial_balance]
        self.reward_memory: list[float] = []
        self.realized_vol_memory: list[float] = []
        self.predicted_vol_memory: list[float] = []
        return self._get_obs(), {}

    def step(self, action: int):
        if self.day >= self.n_steps:
            self.day = self.n_steps - 1
        close_price = float(self.df.loc[self.day, "close"])
        portfolio_value_before = self.cash + self.position * close_price

        if action == 1:
            amount_to_buy = self.cash / (close_price * (1.0 + self.fee_pct))
            if amount_to_buy > 0:
                cost = amount_to_buy * close_price * (1.0 + self.fee_pct)
                if cost <= self.cash:
                    self.cash -= cost
                    self.position += amount_to_buy
        elif action == 2:
            if self.position > 0:
                proceeds = self.position * close_price * (1.0 - self.fee_pct)
                self.cash += proceeds
                self.position = 0.0

        self.day += 1
        if self.day >= self.n_steps:
            self.day = self.n_steps - 1

        new_close_price = float(self.df.loc[self.day, "close"])
        portfolio_value = self.cash + self.position * new_close_price

        ret = (portfolio_value - portfolio_value_before) / (portfolio_value_before + 1e-6)
        realized_vol = float(self.df.loc[self.day, "vol"])
        predicted_vol = self._predicted_vol(self.day)

        if self.reward_type == "sortino":
            window = min(self.day, self.lookback)
            if window > 1:
                rets = pd.Series(self.asset_memory[-window:]).pct_change().dropna()
                downside = rets[rets < 0]
                downside_std = float(downside.std()) if len(downside) > 1 else 1e-6
                reward = float(ret / (downside_std + 1e-6)) * self.reward_scaling
            else:
                reward = float(ret) * self.reward_scaling
        else:
            vol_prediction_error = np.abs(predicted_vol - realized_vol / (0.05 + 1e-6))
            vol_pred_reward = -vol_prediction_error * self.gamma
            reward = (self.alpha * ret - self.beta * realized_vol + vol_pred_reward)
            reward = float(reward * self.reward_scaling)

        self.asset_memory.append(portfolio_value)
        self.position_memory.append(self.position)
        self.cash_memory.append(self.cash)
        self.reward_memory.append(reward)
        self.realized_vol_memory.append(realized_vol)
        self.predicted_vol_memory.append(predicted_vol)

        terminated = self.day >= self.n_steps - 1
        truncated = False
        obs = self._get_obs()
        info = {
            "portfolio_value": portfolio_value,
            "cash": self.cash,
            "position": self.position,
            "close_price": new_close_price,
            "predicted_vol": predicted_vol,
            "realized_vol": realized_vol,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        return None


REWARD_PRESETS = {
    "aggressive": {"reward_alpha": 2.0, "reward_beta": 0.2, "reward_gamma": 0.0, "reward_scaling": 5e-4, "fee_pct": 0.0001},
    "balanced": {"reward_alpha": 1.0, "reward_beta": 2.0, "reward_gamma": 1.0, "reward_scaling": 1e-4, "fee_pct": 0.001},
    "vol_focused": {"reward_alpha": 0.5, "reward_beta": 1.0, "reward_gamma": 2.0, "reward_scaling": 1e-4, "fee_pct": 0.001},
    "conservative": {"reward_alpha": 0.5, "reward_beta": 3.0, "reward_gamma": 1.0, "reward_scaling": 1e-4, "fee_pct": 0.005},
}


def make_env(df: pd.DataFrame, preset: str = "balanced", **kwargs) -> "TradingEnv":
    params = REWARD_PRESETS.get(preset, REWARD_PRESETS["balanced"])
    merged = {**params, **kwargs}
    return TradingEnv(df, **merged)


def train_ppo(env: "TradingEnv", total_timesteps: int = 50_000, lr: float = 3e-4, device: str = "auto"):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", vec_env, learning_rate=lr, verbose=1, device=device)
    model.learn(total_timesteps=total_timesteps)
    return model
