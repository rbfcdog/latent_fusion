import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEQ_LEN = 10
FFILL_LIMIT = 5
TEXT_LAG = 1


def _load_prices(ticker, ts_dir):
    path = f"{ts_dir}/{ticker.lower()}.csv"
    df = pd.read_csv(path, usecols=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def load_panel(cache_dir="cache", ts_dir="data/time_series", seq_len=SEQ_LEN,
               train_frac=0.70, val_frac=0.15, text_lag=TEXT_LAG):
    text_meta = pd.read_csv(f"{cache_dir}/text/top50_daily_metadata.csv")
    text_emb = np.load(f"{cache_dir}/text/top50_daily_embeddings.npy", mmap_mode="r")
    mom_meta = pd.read_csv(f"{cache_dir}/moment/top50_moment_metadata.csv")
    mom_emb = np.load(f"{cache_dir}/moment/top50_moment_embeddings.npy", mmap_mode="r")

    text_meta = text_meta.reset_index().rename(columns={"index": "text_idx"})
    text_meta["date"] = pd.to_datetime(text_meta["date"])
    mom_meta = mom_meta.reset_index().rename(columns={"index": "mom_idx"})
    mom_meta["date"] = pd.to_datetime(mom_meta["date"])

    tickers = sorted(set(text_meta.ticker) & set(mom_meta.ticker))
    arrays = []
    names = []
    for ticker in tickers:
        px = _load_prices(ticker, ts_dir)
        t = text_meta[text_meta.ticker == ticker][["date", "text_idx"]]
        m = mom_meta[mom_meta.ticker == ticker][["date", "mom_idx"]]
        df = px.merge(t, left_on="Date", right_on="date", how="left").drop(columns="date")
        df = df.merge(m, left_on="Date", right_on="date", how="left").drop(columns="date")
        df["text_idx"] = df["text_idx"].shift(text_lag)
        df[["text_idx", "mom_idx"]] = df[["text_idx", "mom_idx"]].ffill(limit=FFILL_LIMIT)
        df["known_ret"] = df["Close"].pct_change()
        df["r_fwd"] = df["known_ret"].shift(-1)
        df = df.dropna(subset=["text_idx", "mom_idx", "r_fwd"]).reset_index(drop=True)
        if len(df) < seq_len + 30:
            continue
        names.append(ticker)
        arrays.append({
            "dates": df["Date"].values,
            "text": np.asarray(text_emb[df["text_idx"].astype(int).values], dtype=np.float32),
            "mom": np.asarray(mom_emb[df["mom_idx"].astype(int).values], dtype=np.float32),
            "known_ret": df["known_ret"].values.astype(np.float32),
            "r_fwd": df["r_fwd"].values.astype(np.float32),
        })

    all_dates = np.sort(np.unique(np.concatenate([a["dates"] for a in arrays])))
    train_end = all_dates[int(len(all_dates) * train_frac)]
    val_start = all_dates[int(len(all_dates) * train_frac * (1 - val_frac))]

    tr_text = np.concatenate([a["text"][a["dates"] <= train_end] for a in arrays])
    tr_mom = np.concatenate([a["mom"][a["dates"] <= train_end] for a in arrays])
    text_mu, text_sd = tr_text.mean(0), tr_text.std(0) + 1e-8
    mom_mu, mom_sd = tr_mom.mean(0), tr_mom.std(0) + 1e-8
    for a in arrays:
        a["text"] = (a["text"] - text_mu) / text_sd
        a["mom"] = (a["mom"] - mom_mu) / mom_sd

    mkt = pd.concat([
        pd.Series(a["known_ret"], index=pd.DatetimeIndex(a["dates"]))
        for a in arrays
    ], axis=1).mean(axis=1)
    mkt_vol = mkt.rolling(20).std()
    vol_median = mkt_vol[mkt_vol.index <= train_end].median()
    regime = (mkt_vol > vol_median).astype(int)

    splits = {"train": [], "val": [], "test": []}
    for ti, a in enumerate(arrays):
        for j in range(seq_len - 1, len(a["dates"])):
            d = a["dates"][j]
            if d <= val_start:
                split = "train"
            elif d <= train_end:
                split = "val"
            else:
                split = "test"
            splits[split].append((ti, j))

    return {
        "arrays": arrays,
        "tickers": names,
        "splits": splits,
        "regime": regime,
        "train_end": train_end,
        "val_start": val_start,
    }


def save_panel(panel, path):
    out = {}
    for i, a in enumerate(panel["arrays"]):
        for k in ("dates", "text", "mom", "known_ret", "r_fwd"):
            out[f"arr_{i}_{k}"] = a[k]
    for split, samples in panel["splits"].items():
        out[f"samples_{split}"] = np.asarray(samples, dtype=np.int32)
    out["tickers"] = np.asarray(panel["tickers"])
    out["regime_dates"] = panel["regime"].index.values
    out["regime_vals"] = panel["regime"].values.astype(np.int8)
    out["bounds"] = np.asarray([panel["val_start"], panel["train_end"]], dtype="datetime64[ns]")
    np.savez_compressed(path, **out)


def load_panel_file(path):
    z = np.load(path, allow_pickle=False)
    tickers = [str(t) for t in z["tickers"]]
    arrays = [
        {k: z[f"arr_{i}_{k}"] for k in ("dates", "text", "mom", "known_ret", "r_fwd")}
        for i in range(len(tickers))
    ]
    splits = {s: [tuple(x) for x in z[f"samples_{s}"]] for s in ("train", "val", "test")}
    regime = pd.Series(z["regime_vals"].astype(int), index=pd.DatetimeIndex(z["regime_dates"]))
    val_start, train_end = z["bounds"]
    return {
        "arrays": arrays,
        "tickers": tickers,
        "splits": splits,
        "regime": regime,
        "train_end": train_end,
        "val_start": val_start,
    }


class FusionDataset(Dataset):
    def __init__(self, panel, split):
        self.arrays = panel["arrays"]
        self.samples = panel["splits"][split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ti, j = self.samples[i]
        a = self.arrays[ti]
        sl = slice(j - SEQ_LEN + 1, j + 1)
        return (
            torch.from_numpy(a["mom"][sl]),
            torch.from_numpy(a["text"][sl]),
            torch.tensor(a["r_fwd"][j], dtype=torch.float32),
        )


def sample_frame(panel, split):
    rows = []
    for ti, j in panel["splits"][split]:
        a = panel["arrays"][ti]
        rows.append({
            "ticker": panel["tickers"][ti],
            "date": a["dates"][j],
            "r_fwd": a["r_fwd"][j],
            "known_ret": a["known_ret"][j],
        })
    df = pd.DataFrame(rows)
    df["regime"] = df["date"].map(panel["regime"]).fillna(0).astype(int)
    return df
