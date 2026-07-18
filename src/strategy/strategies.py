from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None

logger = logging.getLogger(__name__)


@dataclass
class SmaCrossStrategy:
    fast_window: int = 20
    slow_window: int = 100
    price_col: str = "close"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        price = pd.to_numeric(df[self.price_col], errors="coerce")
        fast = price.rolling(self.fast_window).mean()
        slow = price.rolling(self.slow_window).mean()
        signal = (fast > slow).astype(float)
        return signal.fillna(0.0)


@dataclass
class MeanReversionStrategy:
    lookback: int = 50
    z_entry: float = 1.5
    z_exit: float = 0.3
    price_col: str = "close"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        price = pd.to_numeric(df[self.price_col], errors="coerce")
        rolling_mean = price.rolling(self.lookback).mean()
        rolling_std = price.rolling(self.lookback).std()
        z = (price - rolling_mean) / rolling_std

        position = 0.0
        signals: list[float] = []
        for z_val in z.fillna(0.0):
            if position == 0.0:
                if z_val <= -self.z_entry:
                    position = 1.0
                elif z_val >= self.z_entry:
                    position = -1.0
            elif position > 0.0:
                if z_val >= -self.z_exit:
                    position = 0.0
            else:
                if z_val <= self.z_exit:
                    position = 0.0
            signals.append(position)

        return pd.Series(signals, index=df.index, dtype=float)


@dataclass
class HMMRegimeStrategy:
    n_states: int = 3
    warmup: int = 500
    refit_interval: int = 100
    vol_window: int = 24
    n_iter: int = 200
    random_state: int = 42
    train_window: int | None = 50_000

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if GaussianHMM is None:
            raise ImportError("hmmlearn is not installed")

        close = pd.to_numeric(df["close"], errors="coerce")
        log_close = pd.Series(np.log(close.to_numpy(dtype=float)), index=df.index)
        ret = log_close.diff().fillna(0.0)
        vol = ret.rolling(self.vol_window).std().fillna(0.0)

        features = pd.DataFrame({"ret": ret, "vol": vol}, index=df.index)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x = features.to_numpy(dtype=float)

        signals_arr = np.zeros(len(df), dtype=float)
        model: Any = None
        x_mean = np.zeros(x.shape[1], dtype=float)
        x_std = np.ones(x.shape[1], dtype=float)
        state_to_signal: dict[int, float] = {}

        start = max(self.warmup, self.vol_window + 2)
        refit_step = max(self.refit_interval, 1)
        i = start
        while i < len(x):
            should_refit = model is None or ((i - start) % refit_step == 0)
            if should_refit:
                train_start = 0
                if self.train_window is not None and self.train_window > 0:
                    train_start = max(0, i - self.train_window)
                train = x[train_start:i]
                if len(train) < self.n_states * 20:
                    i += refit_step
                    continue

                x_mean = train.mean(axis=0)
                x_std = train.std(axis=0)
                x_std = np.where(x_std == 0.0, 1.0, x_std)
                train_z = (train - x_mean) / x_std

                hmm = GaussianHMM(
                    n_components=self.n_states, covariance_type="full",
                    n_iter=self.n_iter, random_state=self.random_state,
                )
                try:
                    hmm.fit(train_z)
                    hidden_train = hmm.predict(train_z)
                except Exception:
                    model = None
                    i += refit_step
                    continue

                state_ret_means = {
                    s: float(np.mean(train[hidden_train == s, 0]))
                    for s in range(self.n_states) if np.any(hidden_train == s)
                }
                ordered = sorted(state_ret_means.keys(), key=lambda s: state_ret_means[s])
                if len(ordered) < self.n_states:
                    model = None
                    i += refit_step
                    continue

                state_to_signal = {ordered[0]: -1.0, ordered[len(ordered) // 2]: 0.0, ordered[-1]: 1.0}
                model = hmm

            next_i = min(i + refit_step, len(x))
            if model is not None and next_i > i:
                x_block_z = (x[i:next_i] - x_mean) / x_std
                try:
                    states = model.predict(x_block_z)
                    signals_arr[i:next_i] = np.array(
                        [state_to_signal.get(int(s), 0.0) for s in states], dtype=float
                    )
                except Exception:
                    signals_arr[i:next_i] = 0.0
            i = next_i

        return pd.Series(signals_arr, index=df.index, dtype=float).shift(1).fillna(0.0)


@dataclass
class VwapReversionStrategy:
    vwap_col: str = "vwap"
    band_col: str = "vwap_l2"
    upper_band_col: str = "vwap_u2"
    cooldown: int = 10

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=df.index)
        if self.band_col not in df.columns or self.upper_band_col not in df.columns:
            return signal

        long_sig = df["close"] < df[self.band_col]
        short_sig = df["close"] > df[self.upper_band_col]

        last_signal = -self.cooldown - 1
        for i in range(len(df)):
            if i - last_signal < self.cooldown:
                continue
            if long_sig.iloc[i]:
                signal.iloc[i] = 1.0
                last_signal = i
            elif short_sig.iloc[i]:
                signal.iloc[i] = -1.0
                last_signal = i

        return signal


@dataclass
class InstitutionalV3Strategy:
    lookback: int = 20
    recovery_bars: int = 5
    min_wick_ratio: float = 0.0008
    breakout_pct: float = 0.0005
    score_threshold: float = 70.0
    cooldown: int = 20
    vwap_tolerance: float = 1.5

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        lo = df["low"].astype(float).values
        hi = df["high"].astype(float).values
        cl = df["close"].astype(float).values

        lb = self.lookback
        rsl = pd.Series(lo).rolling(lb, min_periods=lb).min().values
        rsh = pd.Series(hi).rolling(lb, min_periods=lb).max().values
        mw = self.min_wick_ratio
        bp = self.breakout_pct
        rb = self.recovery_bars

        liq_l = np.zeros(n, dtype=bool)
        liq_s = np.zeros(n, dtype=bool)
        for i in range(lb, n - rb):
            prev_lo = lo[i - 1]
            prev_rsl = rsl[i - 2] if i >= 2 else rsl[i - 1]
            swl = (lo[i] < rsl[i - 1]) and (prev_lo >= prev_rsl)
            swlv = (rsl[i - 1] - lo[i]) > (rsl[i - 1] * mw)
            rc = np.max(cl[i + 1 : i + rb + 1]) > rsl[i - 1] if i + rb < n else False
            liq_l[i] = swl and swlv and rc

            prev_hi = hi[i - 1]
            prev_rsh = rsh[i - 2] if i >= 2 else rsh[i - 1]
            swh = (hi[i] > rsh[i - 1]) and (prev_hi <= prev_rsh)
            swhv = (hi[i] - rsh[i - 1]) > (rsh[i - 1] * mw)
            rcs = np.min(cl[i + 1 : i + rb + 1]) < rsh[i - 1] if i + rb < n else False
            liq_s[i] = swh and swhv and rcs

        ph = pd.Series(hi).rolling(lb, min_periods=1).max().values
        pl = pd.Series(lo).rolling(lb, min_periods=1).min().values
        msb_b = np.zeros(n, dtype=bool)
        msb_s = np.zeros(n, dtype=bool)
        for i in range(1, n):
            msb_b[i] = (cl[i] > ph[i - 1]) and ((cl[i] - ph[i - 1]) / ph[i - 1] > bp)
            msb_s[i] = (cl[i] < pl[i - 1]) and ((pl[i - 1] - cl[i]) / pl[i - 1] > bp)

        has_liq_l = pd.Series(liq_l).rolling(lb * 2, min_periods=1).max().values > 0
        has_liq_s = pd.Series(liq_s).rolling(lb * 2, min_periods=1).max().values > 0
        msb_bv = msb_b & has_liq_l
        msb_sv = msb_s & has_liq_s

        signal = pd.Series(0.0, index=df.index)
        last_signal = -self.cooldown - 1
        for i in range(lb * 2, n):
            if i - last_signal < self.cooldown:
                continue
            long_score = (
                (30 if has_liq_l[i] else 0)
                + (30 if msb_bv[i] else 0)
                + (20 if abs(df.get("vwap_d", pd.Series(0, index=df.index)).iloc[i]) < self.vwap_tolerance else 0)
            )
            short_score = (
                (30 if has_liq_s[i] else 0)
                + (30 if msb_sv[i] else 0)
                + (20 if abs(df.get("vwap_d", pd.Series(0, index=df.index)).iloc[i]) < self.vwap_tolerance else 0)
            )
            if long_score >= self.score_threshold and cl[i] > cl[i - 1]:
                signal.iloc[i] = 1.0
                last_signal = i
            elif short_score >= self.score_threshold and cl[i] < cl[i - 1]:
                signal.iloc[i] = -1.0
                last_signal = i

        return signal


@dataclass
class RegimeRouterStrategy:
    vol_window: int = 20
    regime_percentile: float = 50.0
    mom_window: int = 5
    rev_window: int = 1
    regime_train_ratio: float = 0.7

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = pd.to_numeric(df["close"], errors="coerce")
        ret = close.pct_change().fillna(0.0)
        vol = ret.rolling(self.vol_window).std()

        n = len(df)
        split = int(n * self.regime_train_ratio)
        vol_train = vol.iloc[:split].dropna()
        if len(vol_train) < 20:
            vol_threshold = vol.median()
        else:
            vol_threshold = vol_train.quantile(self.regime_percentile / 100.0)

        regime = (vol >= vol_threshold).astype(float)

        rev = -np.sign(ret.rolling(self.rev_window).mean())
        mom = np.sign(ret.rolling(self.mom_window).mean())

        signal = regime * rev + (1.0 - regime) * mom
        signal = signal.fillna(0.0).clip(-1.0, 1.0)
        return pd.Series(signal.values, index=df.index)


@dataclass
class IntensityGatedStrategy:
    vol_window: int = 20
    regime_percentile: float = 50.0
    mom_window: int = 5
    rev_window: int = 1
    gate_percentile: float = 70.0
    gate_type: str = "hard"
    regime_train_ratio: float = 0.7

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = pd.to_numeric(df["close"], errors="coerce")
        ret = close.pct_change().fillna(0.0)
        vol = ret.rolling(self.vol_window).std()

        n = len(df)
        split = int(n * self.regime_train_ratio)
        vol_train = vol.iloc[:split].dropna()
        vol_threshold = vol_train.quantile(self.regime_percentile / 100.0) if len(vol_train) >= 20 else vol.median()
        regime = (vol >= vol_threshold).astype(float)

        intensity = vol.rolling(self.vol_window).mean()
        intensity_rank = intensity.rank(pct=True)
        gate_threshold = self.gate_percentile / 100.0
        if self.gate_type == "hard":
            gate = (intensity_rank >= gate_threshold).astype(float)
        else:
            gate = ((intensity_rank - 0.5).clip(lower=0.0) * 2.0).clip(upper=1.0)

        rev = -np.sign(ret.rolling(self.rev_window).mean())
        mom = np.sign(ret.rolling(self.mom_window).mean())
        directional = regime * rev + (1.0 - regime) * mom

        signal = gate * directional
        signal = signal.fillna(0.0).clip(-1.0, 1.0)
        return pd.Series(signal.values, index=df.index)


@dataclass
class S1Hard70Strategy:
    vol_window: int = 20
    regime_percentile: float = 50.0
    mom_window: int = 3
    rev_window: int = 1
    gate_percentile: float = 70.0
    intensity_lookback: int = 252
    regime_train_ratio: float = 0.7

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = pd.to_numeric(df["close"], errors="coerce")
        ret = close.pct_change().fillna(0.0)
        vol = ret.rolling(self.vol_window).std()

        n = len(df)
        split = int(n * self.regime_train_ratio)
        vol_train = vol.iloc[:split].dropna()
        vol_threshold = (
            vol_train.quantile(self.regime_percentile / 100.0)
            if len(vol_train) >= 20
            else vol.median()
        )
        regime = (vol >= vol_threshold).astype(float)

        intensity = vol.ewm(span=self.vol_window, adjust=False).mean()
        int_rank = intensity.rolling(self.intensity_lookback, min_periods=20).apply(
            lambda x: (x < x.iloc[-1]).mean()
        )
        gate_threshold = self.gate_percentile / 100.0
        hard_gate = (int_rank >= gate_threshold).fillna(0.0).astype(float)

        rev = -np.sign(ret.rolling(self.rev_window).mean())
        mom = np.sign(ret.rolling(self.mom_window).mean())
        directional = regime * rev + (1.0 - regime) * mom

        signal = hard_gate * directional
        signal = signal.fillna(0.0).clip(-1.0, 1.0)
        return pd.Series(signal.values, index=df.index)
