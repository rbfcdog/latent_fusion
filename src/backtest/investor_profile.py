from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InvestorProfile:
    name: str
    max_position: float = 1.0
    max_leverage: float = 1.0
    vol_target: float = 0.0
    drawdown_limit: float = 0.0
    signal_clip: float = 1.0
    vol_lookback: int = 20

    def apply_signal(self, signal: float) -> float:
        clipped = float(np.clip(signal, -self.signal_clip, self.signal_clip))
        clipped = float(np.clip(clipped, -self.max_position, self.max_position))
        if self.max_leverage < 1.0:
            clipped *= self.max_leverage
        elif self.max_leverage > 1.0:
            clipped = float(np.clip(clipped * self.max_leverage, -self.max_leverage, self.max_leverage))
        return clipped

    def vol_scale(self, returns: pd.Series, pos: int) -> float:
        if self.vol_target <= 0 or pos < self.vol_lookback:
            return 1.0
        recent = returns.iloc[max(0, pos - self.vol_lookback):pos]
        if len(recent) < 5:
            return 1.0
        realized_vol = float(recent.std())
        if realized_vol < 1e-8:
            return 1.0
        scale = self.vol_target / realized_vol
        return float(np.clip(scale, 0.1, 3.0))

    def drawdown_guard(self, equity: float, peak: float) -> float:
        if self.drawdown_limit <= 0 or peak <= 0:
            return 1.0
        dd = (equity / peak) - 1.0
        if dd < -self.drawdown_limit:
            return 0.0
        return 1.0


CONSERVATIVE = InvestorProfile(
    name="conservative",
    max_position=0.5,
    max_leverage=0.8,
    vol_target=0.10,
    drawdown_limit=0.08,
    signal_clip=0.5,
)

MODERATE = InvestorProfile(
    name="moderate",
    max_position=0.75,
    max_leverage=1.0,
    vol_target=0.15,
    drawdown_limit=0.15,
    signal_clip=0.75,
)

AGGRESSIVE = InvestorProfile(
    name="aggressive",
    max_position=1.0,
    max_leverage=1.5,
    vol_target=0.25,
    drawdown_limit=0.25,
    signal_clip=1.0,
)

PROFILES = {
    "conservative": CONSERVATIVE,
    "moderate": MODERATE,
    "aggressive": AGGRESSIVE,
}
