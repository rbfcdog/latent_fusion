"""Illustrative portfolio policies on saved, out-of-sample fusion predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.fusion.event_experiment import _portfolio, _portfolio_metrics

# Fully invested in the same eight assets; sessions approximate month/quarter.
PROFILES = {
    "conservador": {"sessions": 63, "strength": 0.5, "max_position": 0.18},
    "moderado": {"sessions": 21, "strength": 1.0, "max_position": 0.25},
    "arrojado": {"sessions": 5, "strength": 2.0, "max_position": 0.40},
}


def profile_weights(scores, profile):
    """Clip a monotone signal tilt while distributing the remainder below the cap."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Pontuações inválidas para alocação")
    cap = profile["max_position"]
    if cap * len(scores) < 1 - 1e-12 or profile["strength"] < 0:
        raise ValueError("Teto de posição inviável ou perfil inválido")
    z = (scores - scores.mean()) / max(scores.std(), 1e-6)
    base = np.exp(np.clip(0.35 * profile["strength"] * z, -2, 2))
    weights = np.zeros_like(base)
    free = np.ones(len(base), dtype=bool)
    remaining = 1.0
    while free.any():
        candidate = remaining * base[free] / base[free].sum()
        over = candidate > cap
        if not over.any():
            weights[free] = candidate
            break
        indices = np.flatnonzero(free)[over]
        weights[indices] = cap
        remaining -= cap * len(indices)
        free[indices] = False
    if not np.isclose(weights.sum(), 1) or np.any(weights < 0) or np.any(weights > cap + 1e-12):
        raise ValueError("Carteira não satisfaz as restrições do perfil")
    return weights


def compare_profiles(panel, predictions, end_date, *, cost_bps=10):
    """Backtest all fixed policies against the same buy-and-hold calendar and costs."""
    if not 0 <= cost_bps < 10_000:
        raise ValueError("Custo de transação inválido")
    required = ("ts_only", "metadata", "text_residual")
    if set(predictions) != set(required):
        raise ValueError("Ausência de controle numérico, metadados ou texto")
    dates = predictions["ts_only"]["date"]
    for name in required[1:]:
        if not predictions[name][["date", "ticker"]].equals(predictions["ts_only"][["date", "ticker"]]):
            raise ValueError("Ablações não compartilham o mesmo calendário e universo")
    rows, curves, allocations = [], {}, []
    benchmark = None
    for name, profile in PROFILES.items():
        fn = lambda scores: profile_weights(scores, profile)
        for model in required:
            curve = _portfolio(panel, predictions[model], cost_bps,
                               rebalance_sessions=profile["sessions"],
                               weight_fn=fn, end_date=end_date)
            if benchmark is None:
                benchmark = curve["bh"].copy()
            elif not curve.index.equals(benchmark.index) or not np.allclose(curve["bh"], benchmark):
                raise ValueError("Comparações com calendários ou referência diferentes")
            curves[(name, model)] = curve["strategy"].copy()
            metrics = _portfolio_metrics(curve, cost_bps)
            metrics.update({"perfil": name, "modelo": model,
                            "rebalanceamentos": metrics.pop("rebalance_count"),
                            "queda_máxima_pct": 100 * (curve["strategy"] / np.maximum(1, curve["strategy"].cummax()) - 1).min()})
            rows.append(metrics)
            first = dates.min()
            first_scores = predictions[model].loc[predictions[model]["date"] == first]
            if list(first_scores["ticker"]) != list(panel["tickers"]):
                first_scores = first_scores.set_index("ticker").reindex(panel["tickers"]).reset_index()
            if first_scores["score"].isna().any():
                raise ValueError("Primeiro pregão sem previsão para algum ativo")
            allocations.extend({"perfil": name, "modelo": model, "data": pd.Timestamp(first).date(),
                                "ativo": ticker, "peso_pct": 100 * weight}
                               for ticker, weight in zip(panel["tickers"], fn(first_scores["score"].to_numpy())))
    return (pd.DataFrame(rows).set_index(["perfil", "modelo"]),
            pd.DataFrame(curves, index=benchmark.index),
            pd.DataFrame(allocations).set_index(["perfil", "modelo", "ativo"]), benchmark)
