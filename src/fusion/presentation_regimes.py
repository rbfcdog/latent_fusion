"""Posthoc, close-time market regimes for saved fusion prediction CSVs.

A regime on date t uses that day's completed close-to-close returns. It is
not a pre-close trading signal; next-open decisions can use it. Forward
returns and held-out prediction labels are never HMM inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


REQUIRED_MODELS = ("ts_only", "metadata", "text_residual")


def _market_returns(panel):
    tickers = panel["tickers"]
    arrays = panel["arrays"]
    if not tickers or len(tickers) != len(arrays) or len(set(tickers)) != len(tickers):
        raise ValueError("Market panel needs distinct tickers and matching price arrays")
    columns = []
    for ticker, asset in zip(tickers, arrays):
        dates = pd.DatetimeIndex(asset["dates"])
        returns = np.asarray(asset["known_ret"], dtype=float)
        if len(dates) != len(returns) or dates.hasnans or not dates.is_monotonic_increasing or not dates.is_unique:
            raise ValueError(f"Invalid market dates or close returns for {ticker}")
        columns.append(pd.Series(returns, index=dates, name=ticker))
    market = pd.concat(columns, axis=1).sort_index()
    if market.empty:
        raise ValueError("Empty market calendar")
    # Do not substitute zero for an unavailable close, or change the universe.
    complete = market.notna().all(axis=1) & np.isfinite(market.to_numpy()).all(axis=1)
    if not complete.any():
        raise ValueError("No fully observed market dates")
    market = market.loc[complete.idxmax():]
    if not (market.notna().all(axis=1) & np.isfinite(market.to_numpy()).all(axis=1)).all():
        raise ValueError("Incomplete same-date closes on the shared market calendar")
    return market.mean(axis=1)


def _prediction_rows(panel, predictions):
    if not isinstance(predictions, dict) or not set(REQUIRED_MODELS).issubset(predictions):
        raise ValueError("Missing numeric, metadata or text-residual prediction CSV")
    samples = np.asarray(panel["splits"]["test"])
    if samples.ndim != 2 or samples.shape[1] != 2 or not len(samples):
        raise ValueError("Missing test samples in panel")
    try:
        expected = pd.MultiIndex.from_tuples(
            [(panel["tickers"][int(i)], pd.Timestamp(panel["arrays"][int(i)]["dates"][int(j)]))
             for i, j in samples], names=["ticker", "date"])
    except (IndexError, KeyError) as exc:
        raise ValueError("Invalid test sample index") from exc
    if not expected.is_unique:
        raise ValueError("Duplicate test samples")
    expected = expected.sort_values()
    aligned = {}
    for name, frame in predictions.items():
        if not isinstance(frame, pd.DataFrame) or not {"ticker", "date", "score", "excess_return_5d", "has_event"}.issubset(frame):
            raise ValueError(f"Invalid prediction columns for {name}")
        part = frame.copy()
        part["date"] = pd.to_datetime(part["date"], errors="raise")
        if part["date"].isna().any():
            raise ValueError(f"Missing prediction date for {name}")
        part = part.set_index(["ticker", "date"]).sort_index()
        if not part.index.is_unique or not part.index.equals(expected):
            raise ValueError(f"Prediction calendar differs from panel test samples: {name}")
        for field in ("score", "excess_return_5d"):
            part[field] = pd.to_numeric(part[field], errors="raise")
            if not np.isfinite(part[field].to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite {field} in {name}")
        if part["has_event"].isna().any() or not part["has_event"].isin([True, False, 0, 1]).all():
            raise ValueError(f"Invalid headline coverage in {name}")
        part["has_event"] = part["has_event"].astype(bool)
        aligned[name] = part
    reference = aligned["ts_only"]
    for name, part in aligned.items():
        if not np.array_equal(part["excess_return_5d"], reference["excess_return_5d"]) or not part["has_event"].equals(reference["has_event"]):
            raise ValueError(f"Targets or headline masks differ from numeric predictions: {name}")
    return aligned


def _filter_states(market, train_end, n_states, random_state, min_rows):
    train = market.loc[:train_end]
    if len(train) < max(min_rows, 10 * n_states):
        raise ValueError("Insufficient complete training market dates for the HMM")
    # A fixed unit conversion, rather than a scaler fit using held-out dates.
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200,
                        tol=1e-4, min_covar=1e-4, random_state=random_state)
    model.fit(train.to_numpy(dtype=float).reshape(-1, 1) * 100)
    variance = model.covars_.reshape(n_states, -1)[:, 0]
    order = np.lexsort((np.arange(n_states), variance))
    rank = np.empty(n_states, dtype=int)
    rank[order] = np.arange(n_states)
    means = model.means_[:, 0]
    probabilities = np.log(np.clip(model.startprob_, 1e-300, None))
    transitions = np.log(np.clip(model.transmat_, 1e-300, None))
    records = []
    for day, value in market.items():
        # Forward recursion: p(z_t | close_<=t), never Viterbi/smoothing over test.
        if records:
            probabilities = np.logaddexp.reduce(probabilities[:, None] + transitions, axis=0)
        observation = float(value) * 100
        probabilities += -0.5 * (np.log(2 * np.pi * variance) + (observation - means) ** 2 / variance)
        probabilities -= np.logaddexp.reduce(probabilities)
        state = int(np.argmax(probabilities))
        regime_id = int(rank[state])
        label = ("low_vol", "high_vol")[regime_id] if n_states == 2 else f"vol_{regime_id}"
        records.append((day, regime_id, label, float(np.exp(probabilities[state])),
                        float(value), day <= train_end))
    states = pd.DataFrame(records, columns=["date", "regime_id", "regime", "state_probability",
                                            "market_return_1d", "is_fit_date"])
    states.attrs["hmm_converged"] = bool(model.monitor_.converged)
    states.attrs["train_end"] = train_end
    states.attrs["fit_days"] = len(train)
    states.attrs["train_variance_by_regime"] = (variance[order] / 10_000).tolist()
    return states


def _daily_ic(frame):
    correlations = []
    for _, day in frame.groupby("date"):
        if len(day) >= 3 and day["score"].nunique() > 1 and day["excess_return_5d"].nunique() > 1:
            correlations.append(day["score"].corr(day["excess_return_5d"], method="spearman"))
    return float(np.mean(correlations)) if correlations else np.nan


def analyze_regimes(panel, predictions, *, n_states=2, random_state=42, min_rows=30):
    """Return (per-regime metrics, chronological filtered states) from saved CSV frames.

    HMM fit: equal-weight market close returns on complete dates <= bounds[0]
    (train end), never val/test emissions. Recursion subsequently conditions on
    known closes only, through each test date. No target, forward-return column,
    headline, score or test label enters fitting/filtering. Predictions must
    match panel test samples exactly; extra aligned control models are included.
    Scores/labels are read *only* for posthoc performance reporting.
    IC is mean daily cross-sectional Spearman correlation. A positive
    mse_gain_vs_numeric means smaller model error; conditional_mse_gain
    subtracts the gain on rows without news from the gain on news rows.
    Neither is a causal treatment effect. Small cohorts and overlapping
    five-day targets make these descriptive, unstable estimates.
    """
    if not isinstance(n_states, int) or n_states < 2 or not isinstance(min_rows, int) or min_rows < 1:
        raise ValueError("Invalid HMM state count or training minimum")
    bounds = pd.DatetimeIndex(panel["bounds"])
    if len(bounds) != 2 or bounds.hasnans or bounds[0] >= bounds[1]:
        raise ValueError("Invalid chronological train/validation bounds")
    aligned = _prediction_rows(panel, predictions)
    first_test = aligned["ts_only"].index.get_level_values("date").min()
    last_test = aligned["ts_only"].index.get_level_values("date").max()
    if first_test <= bounds[1] or bounds[0] >= first_test:
        raise ValueError("Test dates must be strictly after train and validation")
    market = _market_returns(panel).loc[:last_test]
    if last_test not in market.index or first_test not in market.index:
        raise ValueError("Test dates absent from shared close calendar")
    states = _filter_states(market, bounds[0], n_states, random_state, min_rows)
    state_on_date = states.set_index("date")["regime_id"]
    test_dates = aligned["ts_only"].index.get_level_values("date")
    regimes = state_on_date.reindex(test_dates)
    if regimes.isna().any():
        raise ValueError("Test dates without filtered market regimes")
    rows = []
    baseline = aligned["ts_only"]
    for regime_id in range(n_states):
        selected = regimes.to_numpy() == regime_id
        reference = baseline.iloc[selected].reset_index()
        n = len(reference)
        n_dates = reference["date"].nunique()
        news_count = int(reference["has_event"].sum())
        state_days = states.loc[states["regime_id"].eq(regime_id) & states["is_fit_date"]]
        posterior = states.loc[states["date"].isin(reference["date"]), "state_probability"]
        row = {"regime_id": regime_id,
               "regime": ("low_vol", "high_vol")[regime_id] if n_states == 2 else f"vol_{regime_id}",
               "n_dates": n_dates, "n": n, "n_with_news": news_count,
               "headline_coverage": news_count / n if n else np.nan,
               "train_end": states.attrs["train_end"], "n_train_dates": states.attrs["fit_days"],
               "n_train_state_days": len(state_days),
               "train_state_variance": states.attrs["train_variance_by_regime"][regime_id],
               "hmm_converged": states.attrs["hmm_converged"],
               "mean_state_probability": float(posterior.mean()) if n_dates else np.nan,
               "caution": "exploratory, overlapping 5d labels; " +
                          ("few dates/rows; " if n_dates < min_rows or n < min_rows else "") +
                          ("few train-state days; " if len(state_days) < min_rows else "") +
                          ("few news/no-news rows; " if min(news_count, n - news_count) < min_rows else "") +
                          ("uncertain state assignments; " if n_dates and posterior.mean() < 0.7 else "") +
                          ("HMM not converged; " if not states.attrs["hmm_converged"] else "") +
                          "regime known only after close"}
        for name, frame in aligned.items():
            subset = frame.iloc[selected].reset_index()
            errors = (subset["score"] - subset["excess_return_5d"]) ** 2
            row[f"mse_{name}"] = float(errors.mean()) if n else np.nan
            row[f"mse_news_{name}"] = float(errors[subset["has_event"]].mean()) if news_count else np.nan
            row[f"mse_no_news_{name}"] = float(errors[~subset["has_event"]].mean()) if n > news_count else np.nan
            row[f"ic_{name}"] = _daily_ic(subset)
        for name in aligned:
            if name != "ts_only":
                row[f"mse_gain_vs_numeric_{name}"] = row["mse_ts_only"] - row[f"mse_{name}"]
                row[f"mse_gain_news_{name}"] = row["mse_news_ts_only"] - row[f"mse_news_{name}"]
                row[f"mse_gain_no_news_{name}"] = row["mse_no_news_ts_only"] - row[f"mse_no_news_{name}"]
                row[f"conditional_mse_gain_{name}"] = row[f"mse_gain_news_{name}"] - row[f"mse_gain_no_news_{name}"]
                row[f"ic_change_vs_numeric_{name}"] = row[f"ic_{name}"] - row["ic_ts_only"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("regime_id"), states
