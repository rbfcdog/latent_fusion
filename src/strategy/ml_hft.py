import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

class MLHFTStrategy:
    def __init__(self, lookback=60, retrain_every=20, model_type="sgd"):
        self.lookback = lookback
        self.retrain_every = retrain_every
        self.model_type = model_type
        self.model = SGDRegressor(loss="huber", penalty="l2", alpha=0.0001,
                                  learning_rate="adaptive", eta0=0.01,
                                  random_state=42)
        self.scaler = StandardScaler()
        self._trained = False
        self._step = 0
        self._feature_names = None

    def _compute_features(self, df):
        close = df["close"].values
        volume = df["volume"].values
        high = df["high"].values
        low = df["low"].values

        features = {}

        for horizon in [1, 3, 5, 10, 20]:
            if len(close) > horizon:
                features[f"ret_{horizon}"] = np.append([0] * horizon, np.diff(close, horizon) / (close[:-horizon] + 1e-12))
            else:
                features[f"ret_{horizon}"] = np.zeros(len(close))

        if len(close) >= 14:
            delta = pd.Series(np.diff(close, prepend=close[0]))
            gains = delta.clip(lower=0)
            losses = (-delta).clip(lower=0)
            avg_gain = gains.rolling(14, min_periods=1).mean().values
            avg_loss = losses.rolling(14, min_periods=1).mean().values
            rs = np.where(avg_loss > 1e-12, avg_gain / avg_loss, 100.0)
            features["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        else:
            features["rsi_14"] = np.full(len(close), 50.0)

        if len(close) >= 20:
            features["vol_20"] = pd.Series(close).pct_change().rolling(20).std().fillna(0).values
        else:
            features["vol_20"] = np.zeros(len(close))

        if len(volume) >= 10:
            ma10 = pd.Series(volume).rolling(10, min_periods=1).mean().values
            features["vol_ma_ratio"] = np.where(ma10 > 0, volume / ma10, 1.0)
        else:
            features["vol_ma_ratio"] = np.ones(len(close))
        if len(close) >= 5:
            features["hl_position"] = np.where(high - low > 0, (close - low) / (high - low + 1e-12), 0.5)
            features["range_pct"] = np.where(close > 0, (high - low) / close, 0)
        else:
            features["hl_position"] = np.full(len(close), 0.5)
            features["range_pct"] = np.zeros(len(close))

        if "vwap" in df.columns and not df["vwap"].isna().all():
            vwap = df["vwap"].fillna(close).values
            features["vwap_dist"] = np.where(vwap > 0, (close - vwap) / vwap, 0)
        else:
            features["vwap_dist"] = np.zeros(len(close))

        features["price_level"] = np.where(close > 0, close / close[-1] if close[-1] > 0 else np.ones(len(close)), np.ones(len(close)))

        return pd.DataFrame(features)

    def _compute_target(self, df):
        close = df["close"].values
        if len(close) < 2:
            return np.zeros(len(close))
        target = np.append(np.diff(close) / (close[:-1] + 1e-12), [0])
        return np.clip(target, -0.05, 0.05)

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

        if len(recent_X) < 10:
            return pd.Series(0.0, index=df.index)

        self._step += 1

        if self._step % self.retrain_every == 0 or not self._trained:
            try:
                X_scaled = self.scaler.fit_transform(recent_X)
                self.model.partial_fit(X_scaled, recent_y)
                self._trained = True
            except Exception:
                return pd.Series(0.0, index=df.index)

        try:
            last_X = self.scaler.transform(recent_X[-1:])
            pred = float(self.model.predict(last_X)[0])
        except Exception:
            pred = 0.0

        pred = float(np.clip(pred, -0.02, 0.02))

        signal = np.zeros(len(df))
        signal[-1] = np.sign(pred) if abs(pred) > 0.0001 else 0.0

        return pd.Series(signal, index=df.index, dtype=float)
