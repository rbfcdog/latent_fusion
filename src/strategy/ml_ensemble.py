import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

class MLEnsembleStrategy:
    def __init__(self, lookback=80, retrain_every=20):
        self.lookback = lookback
        self.retrain_every = retrain_every
        self.model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.001,
                                   learning_rate="adaptive", eta0=0.01,
                                   random_state=42)
        self.scaler = StandardScaler()
        self._trained = False
        self._step = 0

    def _compute_features(self, df):
        close = df["close"].values
        volume = df["volume"].values
        high = df["high"].values
        low = df["low"].values
        n = len(close)
        feats = {}

        sma_fast = pd.Series(close).rolling(20, min_periods=1).mean().values
        sma_slow = pd.Series(close).rolling(50, min_periods=1).mean().values
        feats["sma_cross"] = np.where(sma_slow > 0, (sma_fast - sma_slow) / sma_slow, 0)

        delta = pd.Series(np.diff(close, prepend=close[0]))
        gains = delta.clip(lower=0)
        losses = (-delta).clip(lower=0)
        avg_gain = gains.rolling(14, min_periods=1).mean().values
        avg_loss = losses.rolling(14, min_periods=1).mean().values
        rs = np.where(avg_loss > 1e-12, avg_gain / avg_loss, 100.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        feats["rsi"] = (rsi - 50.0) / 50.0

        for h in [1, 3, 5, 10]:
            if n > h:
                ret = np.zeros(n)
                ret[h:] = np.diff(close, h) / (close[:-h] + 1e-12)
                feats[f"ret_{h}"] = np.clip(ret, -0.05, 0.05) * 100
            else:
                feats[f"ret_{h}"] = np.zeros(n)

        hl_range = np.where(high - low > 1e-12, high - low, 1e-12)
        feats["hl_position"] = (close - low) / hl_range
        feats["range_pct"] = np.where(close > 0, (high - low) / close, 0) * 100

        vol_10 = pd.Series(close).pct_change().rolling(10, min_periods=2).std().fillna(0).values
        vol_30 = pd.Series(close).pct_change().rolling(30, min_periods=2).std().fillna(0).values
        feats["vol_regime"] = np.where(vol_30 > 1e-12, vol_10 / vol_30, 1.0)

        if "vwap" in df.columns and not df["vwap"].isna().all():
            vwap = df["vwap"].fillna(close).values
            feats["vwap_dist"] = np.where(vwap > 0, (close - vwap) / vwap, 0) * 100
        else:
            feats["vwap_dist"] = np.zeros(n)

        vol_ma_20 = pd.Series(volume).rolling(20, min_periods=1).mean().values
        feats["vol_spike"] = np.where(vol_ma_20 > 0, volume / vol_ma_20, 1.0)

        return pd.DataFrame(feats)

    def _compute_target(self, df, horizon=3):
        close = df["close"].values
        n = len(close)
        if n <= horizon:
            return np.zeros(n)
        fwd = np.zeros(n)
        fwd[:-horizon] = (close[horizon:] - close[:-horizon]) / (close[:-horizon] + 1e-12)
        return (fwd > 0.0005).astype(int)

    def generate_signals(self, df):
        if len(df) < self.lookback:
            return pd.Series(0.0, index=df.index)

        features = self._compute_features(df)
        target = self._compute_target(df)

        recent_X = features.values[-self.lookback:]
        recent_y = target[-self.lookback:]

        valid = np.isfinite(recent_X).all(axis=1) & np.isfinite(recent_y)
        recent_X = recent_X[valid]
        recent_y = recent_y[valid]

        if len(recent_X) < 20:
            return pd.Series(0.0, index=df.index)

        self._step += 1

        if self._step % self.retrain_every == 0 or not self._trained:
            try:
                X_scaled = self.scaler.fit_transform(recent_X)
                self.model.partial_fit(X_scaled, recent_y, classes=np.array([0, 1]))
                self._trained = True
            except Exception:
                return pd.Series(0.0, index=df.index)

        try:
            last_X = self.scaler.transform(recent_X[-1:])
            proba = float(self.model.predict_proba(last_X)[0, 1])
        except Exception:
            proba = 0.5

        signal = np.zeros(len(df))
        signal[-1] = (proba - 0.5) * 2.0
        signal[-1] = float(np.clip(signal[-1], -1.0, 1.0))

        return pd.Series(signal, index=df.index, dtype=float)
