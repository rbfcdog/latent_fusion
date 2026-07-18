from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrainedEmbeddingStrategy:
    model_path: str = ""
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

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "scaler": self._scaler, "pca": self._pca}, f)

    def fit(self, X, y, model_type="ridge", pca_dim=32):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        self._scaler = StandardScaler()
        X_s = self._scaler.fit_transform(X)

        if pca_dim and pca_dim < X_s.shape[1]:
            self._pca = PCA(n_components=pca_dim)
            X_s = self._pca.fit_transform(X_s)
        else:
            self._pca = None

        if model_type == "ridge":
            from sklearn.linear_model import Ridge
            self._model = Ridge(alpha=1.0)
        elif model_type == "lasso":
            from sklearn.linear_model import Lasso
            self._model = Lasso(alpha=0.001, max_iter=5000)
        elif model_type == "elasticnet":
            from sklearn.linear_model import ElasticNet
            self._model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)
        elif model_type == "xgboost":
            from xgboost import XGBRegressor
            self._model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, verbosity=0)
        elif model_type == "mlp":
            from sklearn.neural_network import MLPRegressor
            self._model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self._model.fit(X_s, y)

    def predict(self, X):
        if self._model is None or self._scaler is None:
            return np.zeros(len(X))
        X_s = self._scaler.transform(X)
        if self._pca is not None:
            X_s = self._pca.transform(X_s)
        return self._model.predict(X_s)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if not emb_cols or self._model is None:
            return pd.Series(0.0, index=df.index)
        X = np.nan_to_num(df[emb_cols].values, 0.0)
        preds = np.nan_to_num(self.predict(X), 0.0)
        return pd.Series(np.clip(preds, -0.5, 0.5), index=df.index[:len(preds)])
