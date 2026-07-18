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
    use_softmax_tilt: bool = True
    _model: Any = field(default=None, repr=False)
    _scaler: Any = field(default=None, repr=False)
    _pca: Any = field(default=None, repr=False)
    _embeddings_cache: dict = field(default_factory=dict, repr=False)

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

    def fit(self, X, y, pca_dim=32):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        if pca_dim and pca_dim < X_scaled.shape[1]:
            self._pca = PCA(n_components=pca_dim)
            X_scaled = self._pca.fit_transform(X_scaled)

        self._model = Ridge(alpha=1.0)
        self._model.fit(X_scaled, y)

    def predict(self, X):
        if self._model is None:
            return np.zeros(len(X))
        if self._scaler is None:
            return np.zeros(len(X))
        X_s = self._scaler.transform(X)
        if self._pca is not None:
            X_s = self._pca.transform(X_s)
        return self._model.predict(X_s)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            return pd.Series(0.0, index=df.index)

        if "emb_0" not in df.columns and self._embeddings_cache:
            emb_cols = [c for c in df.columns if c.startswith("emb_")]
            if not emb_cols:
                return pd.Series(0.0, index=df.index)
            X = df[emb_cols].values
        elif "emb_0" in df.columns:
            emb_cols = [c for c in df.columns if c.startswith("emb_")]
            X = df[emb_cols].values
        else:
            return pd.Series(0.0, index=df.index)

        X = np.nan_to_num(X, 0.0)
        preds = self.predict(X)
        preds = np.nan_to_num(preds, 0.0)

        signals = np.clip(preds, -1.0, 1.0)
        return pd.Series(signals, index=df.index[:len(signals)])
