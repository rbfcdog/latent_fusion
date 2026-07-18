from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_path(data_dir: str, freq: str, split: str) -> Path:
    base = Path(data_dir)
    candidates = [
        base / f"M4_{freq}-{split}.csv",
        base / f"{freq}-{split}.csv",
        base / f"{freq}_{split}.csv",
        base / f"{freq}-train.csv" if split == "train" else None,
    ]
    for cand in candidates:
        if cand is not None and cand.exists():
            return cand
    primary = base / f"M4_{freq}-{split}.csv"
    return primary


def load_m4_dataset(
    freq: str = "Daily",
    split: str = "train",
    data_dir: str = "data/m4",
) -> pd.DataFrame:
    path = _resolve_path(data_dir, freq, split)
    df = pd.read_csv(path)
    if "V1" in df.columns:
        df = df.rename(columns={"V1": "series_id"})
    elif "series_id" not in df.columns:
        df.insert(0, "series_id", [f"S{i}" for i in range(len(df))])

    value_cols = [c for c in df.columns if c != "series_id"]
    df = df[["series_id"] + value_cols]
    return df


def m4_to_long(df: pd.DataFrame) -> dict[str, pd.Series]:
    value_cols = [c for c in df.columns if c != "series_id"]
    out: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        sid = str(row["series_id"])
        values = pd.to_numeric(row[value_cols], errors="coerce").dropna()
        if values.empty:
            continue
        out[sid] = pd.Series(values.to_numpy(dtype=float))
    return out


def m4_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    seasonal_period: int = 1,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    n = min(actual.size, predicted.size)
    actual = actual[:n]
    predicted = predicted[:n]

    if n == 0:
        return {"MASE": float("nan"), "sMAPE": float("nan"), "OWA": float("nan")}

    errors = np.abs(actual - predicted)

    if n > seasonal_period:
        denom_series = np.abs(
            actual[seasonal_period:] - actual[:-seasonal_period]
        )
        naive_mae = float(np.mean(denom_series)) if denom_series.size else 0.0
    else:
        naive_mae = 0.0

    if naive_mae > 0.0:
        mase = float(np.mean(errors) / naive_mae)
    else:
        mase = float("nan")

    denom = np.abs(actual) + np.abs(predicted)
    valid = denom > 0.0
    if np.any(valid):
        smape = float(np.mean(2.0 * errors[valid] / denom[valid]) * 100.0)
    else:
        smape = float("nan")

    owa = float("nan")
    if not np.isnan(mase) and not np.isnan(smape):
        owa = float(0.5 * (mase + smape / 100.0))

    return {"MASE": mase, "sMAPE": smape, "OWA": owa}
