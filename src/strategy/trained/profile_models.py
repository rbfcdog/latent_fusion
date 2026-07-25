import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


@dataclass
class TrainedProfileStrategy:
    model_path: str = ""
    profile_name: str = "moderado"
    _model: Any = field(default=None, repr=False)
    _scaler: Any = field(default=None, repr=False)
    _pca: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self.model_path and Path(self.model_path).exists():
            self._load(self.model_path)

    def _load(self, path):
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self._model = bundle.get("model")
        self._scaler = bundle.get("scaler")
        self._pca = bundle.get("pca")
        self.profile_name = bundle.get("profile_name", self.profile_name)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "scaler": self._scaler,
                "pca": self._pca,
                "profile_name": self.profile_name,
            }, f)

    def fit(self, X, y, model_type="ridge", pca_dim=32, profile_params=None):
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.neural_network import MLPRegressor

        self._scaler = StandardScaler()
        X_s = self._scaler.fit_transform(X)
        X_s = np.nan_to_num(X_s, 0.0)

        if pca_dim and pca_dim < X_s.shape[1]:
            self._pca = PCA(n_components=pca_dim)
            X_s = self._pca.fit_transform(X_s)

        params = profile_params or _default_params(self.profile_name)

        if isinstance(y, pd.Series):
            y = y.values
        y = np.nan_to_num(y.ravel(), 0.0)

        sw = _compute_sample_weights(X_s, y, params)

        if params["use_torch"] and model_type == "mlp":
            self._model = _train_profile_mlp(X_s, y, sw, params)
        elif model_type in ("ridge", "lasso", "elasticnet"):
            alpha = params["alpha"]
            if model_type == "ridge":
                self._model = Ridge(alpha=alpha)
                self._model.fit(X_s, y, sample_weight=sw)
            elif model_type == "lasso":
                self._model = Lasso(alpha=alpha * 5, max_iter=5000)
                self._model.fit(X_s, y)
                self._model = Lasso(alpha=alpha, max_iter=5000)
                self._model.fit(X_s, y)
            elif model_type == "elasticnet":
                self._model = ElasticNet(alpha=alpha, l1_ratio=params.get("l1_ratio", 0.5), max_iter=5000)
                self._model.fit(X_s, y)
        elif model_type == "mlp":
            self._model = MLPRegressor(
                hidden_layer_sizes=tuple(params.get("hidden_layers", (64, 32))),
                alpha=params["alpha"],
                max_iter=params.get("max_iter", 500),
                early_stopping=True,
                random_state=42,
            )
            self._model.fit(X_s, y)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def predict(self, X):
        if self._model is None or self._scaler is None:
            return np.zeros(len(X))
        X_s = self._scaler.transform(X)
        if self._pca is not None:
            X_s = self._pca.transform(X_s)
        if hasattr(self._model, "predict_proba") or "torch" in str(type(self._model)):
            pred = self._model.predict(X_s)
        else:
            pred = self._model.predict(X_s)
        params = _default_params(self.profile_name)
        return np.clip(pred, -params["signal_clip"], params["signal_clip"])

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if not emb_cols or self._model is None:
            return pd.Series(0.0, index=df.index)
        X = np.nan_to_num(df[emb_cols].values, 0.0)
        preds = self.predict(X)
        return pd.Series(preds, index=df.index[:len(preds)])


def _default_params(profile_name):
    base = {
        "signal_clip": 1.0,
        "alpha": 0.001,
        "l1_ratio": 0.5,
        "vol_target": 0.25,
        "drawdown_penalty": 0.0,
        "turnover_penalty": 0.0,
        "size_penalty": 0.0,
        "direction_bonus": 0.0,
        "inactivity_penalty": 0.0,
        "use_torch": False,
        "hidden_layers": (64, 32),
        "max_iter": 500,
    }
    if profile_name == "conservador":
        base.update({
            "signal_clip": 0.50,
            "alpha": 5.0,
            "l1_ratio": 0.7,
            "vol_target": 0.10,
            "drawdown_penalty": 0.50,
            "turnover_penalty": 0.30,
            "size_penalty": 0.15,
        })
    elif profile_name == "moderado":
        base.update({
            "signal_clip": 0.75,
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "vol_target": 0.15,
            "drawdown_penalty": 0.15,
            "turnover_penalty": 0.10,
            "size_penalty": 0.05,
        })
    elif profile_name == "arrojado":
        base.update({
            "signal_clip": 1.00,
            "alpha": 0.0001,
            "l1_ratio": 0.3,
            "vol_target": 0.25,
            "drawdown_penalty": 0.0,
            "turnover_penalty": 0.0,
            "size_penalty": 0.0,
            "direction_bonus": 0.10,
            "inactivity_penalty": 0.05,
            "use_torch": True,
        })
    return base


def _compute_sample_weights(X, y, params):
    n = len(y)
    weights = np.ones(n)

    if params["drawdown_penalty"] > 0:
        rolling_max = np.maximum.accumulate(np.maximum(0, y))
        dds = np.maximum(0, rolling_max - y) / (np.abs(y).mean() + 1e-8)
        weights -= params["drawdown_penalty"] * np.clip(dds / (dds.max() + 1e-8), 0, 1)

    trend = pd.Series(y).rolling(20, min_periods=5).mean().fillna(0).values
    if params["direction_bonus"] > 0:
        trend_weight = 1.0 + params["direction_bonus"] * np.abs(trend) / (np.abs(trend).max() + 1e-8)
        weights *= trend_weight

    lower = {"conservador": 0.6, "moderado": 0.7, "arrojado": 0.5}
    weights = np.clip(weights, lower.get(params.get("_profile", "moderado"), 0.5), 2.0)
    weights = np.nan_to_num(weights, 1.0)
    return weights.astype(np.float32)


class ProfileMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=(64, 32), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PickleableProfileMLP:
    def __init__(self, input_dim, hidden_layers=(64, 32), alpha=0.001):
        self.model = ProfileMLP(input_dim, hidden_layers)
        self.alpha = alpha
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.X_mean = None
        self.X_std = None

    def predict(self, X_np):
        if self.X_mean is None:
            return np.zeros(len(X_np))
        self.model.eval()
        X_n = (X_np - self.X_mean) / (self.X_std + 1e-8)
        with torch.no_grad():
            device = next(self.model.parameters()).device
            X_t = torch.tensor(np.nan_to_num(X_n, 0.0), dtype=torch.float32).to(device)
            return self.model(X_t).cpu().numpy()
    def state_dict(self):
        return {
            "model_state": self.model.state_dict(),
            "alpha": self.alpha,
            "hidden_layers": self.hidden_layers,
            "input_dim": self.input_dim,
            "X_mean": self.X_mean,
            "X_std": self.X_std,
        }

    def load_state_dict(self, d):
        self.model.load_state_dict(d["model_state"])
        self.alpha = d["alpha"]
        self.X_mean = d.get("X_mean")
        self.X_std = d.get("X_std")

def _train_profile_mlp(X, y, sample_weights, params, epochs=100, lr=1e-3, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = PickleableProfileMLP(X.shape[1], params.get("hidden_layers", (64, 32)), alpha=params["alpha"])
    model = wrapper.model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=params["alpha"])

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)
    sw_t = torch.tensor(sample_weights, dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(X_t, y_t, sw_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for bx, by, bw in loader:
            optimizer.zero_grad()
            pred = model(bx)
            mse = ((pred - by) ** 2 * bw).mean()
            loss = mse
            if params["size_penalty"] > 0:
                loss += params["size_penalty"] * torch.mean(torch.abs(pred))
            if params["vol_target"] > 0:
                pred_vol = torch.std(pred) + 1e-8
                vol_gap = (pred_vol - params["vol_target"]) ** 2
                loss += 0.1 * vol_gap
            if params["inactivity_penalty"] > 0:
                inactive = (torch.abs(pred) < 0.001).float().mean()
                loss += params["inactivity_penalty"] * inactive
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    wrapper.X_mean = X.mean(axis=0)
    wrapper.X_std = X.std(axis=0) + 1e-8
    return wrapper
