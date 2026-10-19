import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEQ_LEN = 10
HORIZON = 20
MOMENT_MAX_AGE = 5
PANEL_VERSION = 3


def _load_prices(ticker, ts_dir):
    df = pd.read_csv(f"{ts_dir}/{ticker.lower()}.csv", usecols=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    return (df.dropna(subset=["Date", "Close"])
              .loc[lambda x: x["Close"] > 0]
              .sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True))


def load_panel(cache_dir="cache", ts_dir="data/time_series", seq_len=SEQ_LEN,
               horizon=HORIZON, train_frac=0.70, val_frac=0.15):
    text_meta = pd.read_csv(f"{cache_dir}/text/top50_daily_metadata.csv")
    text_emb = np.load(f"{cache_dir}/text/top50_daily_embeddings.npy", mmap_mode="r")
    mom_meta = pd.read_csv(f"{cache_dir}/moment/top50_moment_metadata.csv")
    mom_emb = np.load(f"{cache_dir}/moment/top50_moment_embeddings.npy", mmap_mode="r")
    text_meta = text_meta.reset_index().rename(columns={"index": "text_idx"})
    mom_meta = mom_meta.reset_index().rename(columns={"index": "mom_idx"})
    text_meta["date"] = pd.to_datetime(text_meta["date"])
    mom_meta["date"] = pd.to_datetime(mom_meta["date"])

    arrays, names = [], []
    for ticker in sorted(set(text_meta.ticker) & set(mom_meta.ticker)):
        px = _load_prices(ticker, ts_dir)
        log_ret = np.log(px["Close"]).diff()
        past_sq = log_ret.pow(2).rolling(horizon, min_periods=horizon).mean()
        future_sq = log_ret.pow(2).shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]
        px["past_vol"] = np.sqrt(252 * past_sq)
        px["target_vol"] = np.sqrt(252 * future_sq / horizon)
        px["target_end"] = px["Date"].shift(-horizon)
        px["known_ret"] = px["Close"].pct_change()

        t = text_meta.loc[text_meta.ticker == ticker, ["date", "text_idx"]].copy()
        trading_dates = px["Date"].to_numpy()
        next_trade = np.searchsorted(trading_dates, t["date"].to_numpy())
        t = t.loc[next_trade < len(px)].copy()
        t["Date"] = trading_dates[next_trade[next_trade < len(px)]]
        t = t.groupby("Date")["text_idx"].agg(list).reset_index()
        m = mom_meta.loc[mom_meta.ticker == ticker, ["date", "mom_idx"]]
        df = px.merge(t, on="Date", how="left")
        df = df.merge(m, left_on="Date", right_on="date", how="left").drop(columns="date")
        df["text_idx"] = df["text_idx"].shift(1)
        df["mom_idx"] = df["mom_idx"].ffill(limit=MOMENT_MAX_AGE)
        df = df.dropna(subset=["mom_idx", "past_vol", "target_vol", "target_end"]).reset_index(drop=True)
        if len(df) < seq_len + 30:
            continue
        names.append(ticker)
        observed = df["text_idx"].notna().to_numpy()
        dim = text_emb.shape[1]
        text = np.zeros((len(df), 2 * dim + 1), dtype=np.float32)
        previous = None
        for row in np.flatnonzero(observed):
            current = np.asarray(text_emb[df.at[row, "text_idx"]], dtype=np.float32).mean(axis=0)
            text[row, :dim] = current
            if previous is not None:
                text[row, dim:2 * dim] = current - previous
                denom = np.linalg.norm(current) * np.linalg.norm(previous)
                if denom > 0:
                    text[row, -1] = 1 - np.dot(current, previous) / denom
            previous = current
        arrays.append({
            "dates": df["Date"].to_numpy(),
            "target_end": df["target_end"].to_numpy(),
            "text": text,
            "text_mask": observed,
            "mom": np.asarray(mom_emb[df["mom_idx"].astype(int).to_numpy()], dtype=np.float32),
            "past_vol": df["past_vol"].to_numpy(dtype=np.float32),
            "target_vol": df["target_vol"].to_numpy(dtype=np.float32),
            "known_ret": df["known_ret"].to_numpy(dtype=np.float32),
        })

    if not arrays:
        raise ValueError("Nenhum ticker com preços e embeddings alinhados")
    all_dates = np.sort(np.unique(np.concatenate([a["dates"] for a in arrays])))
    train_end = all_dates[int(len(all_dates) * train_frac) - 1]
    val_start = all_dates[int(len(all_dates) * train_frac * (1 - val_frac)) - 1]

    tr_text = np.concatenate([a["text"][(a["dates"] <= val_start) & a["text_mask"]] for a in arrays])
    tr_mom = np.concatenate([a["mom"][a["dates"] <= val_start] for a in arrays])
    if not len(tr_text) or not len(tr_mom):
        raise ValueError("Dados de treino insuficientes para padronização")
    text_mu, text_sd = tr_text.mean(0), tr_text.std(0) + 1e-6
    mom_mu, mom_sd = tr_mom.mean(0), tr_mom.std(0) + 1e-6
    for a in arrays:
        a["text"][a["text_mask"]] = (a["text"][a["text_mask"]] - text_mu) / text_sd
        a["mom"] = (a["mom"] - mom_mu) / mom_sd

    mkt = pd.concat([
        pd.Series(a["known_ret"], index=pd.DatetimeIndex(a["dates"])) for a in arrays
    ], axis=1).mean(axis=1)
    mkt_vol = mkt.rolling(horizon, min_periods=horizon).std()
    vol_median = mkt_vol[mkt_vol.index <= val_start].median()
    regime = (mkt_vol > vol_median).astype(int)

    splits = {"train": [], "val": [], "test": []}
    for ti, a in enumerate(arrays):
        for j in range(seq_len - 1, len(a["dates"])):
            d, label_end = a["dates"][j], a["target_end"][j]
            if d <= val_start:
                if label_end <= val_start:
                    splits["train"].append((ti, j))
            elif d <= train_end:
                if label_end <= train_end:
                    splits["val"].append((ti, j))
            else:
                splits["test"].append((ti, j))
    if any(not samples for samples in splits.values()):
        raise ValueError("Split temporal vazio após purga do horizonte")
    return {
        "arrays": arrays, "tickers": names, "splits": splits, "regime": regime,
        "train_end": train_end, "val_start": val_start, "seq_len": seq_len,
        "horizon": horizon,
    }


def save_panel(panel, path):
    out = {"version": np.array(PANEL_VERSION), "tickers": np.asarray(panel["tickers"]),
           "regime_dates": panel["regime"].index.values,
           "regime_vals": panel["regime"].to_numpy(dtype=np.int8),
           "bounds": np.asarray([panel["val_start"], panel["train_end"]], dtype="datetime64[ns]"),
           "seq_len": np.array(panel["seq_len"]), "horizon": np.array(panel["horizon"])}
    for i, a in enumerate(panel["arrays"]):
        for k, value in a.items():
            out[f"arr_{i}_{k}"] = value
    for split, samples in panel["splits"].items():
        out[f"samples_{split}"] = np.asarray(samples, dtype=np.int32)
    np.savez_compressed(path, **out)


def load_panel_file(path):
    with np.load(path, allow_pickle=False) as z:
        if "version" not in z or int(z["version"]) != PANEL_VERSION:
            raise ValueError("Painel obsoleto; gere fusion_vol_panel.npz novamente")
        tickers = z["tickers"].tolist()
        fields = ("dates", "target_end", "text", "text_mask", "mom", "past_vol", "target_vol", "known_ret")
        arrays = [{k: z[f"arr_{i}_{k}"] for k in fields} for i in range(len(tickers))]
        splits = {s: [tuple(x) for x in z[f"samples_{s}"]] for s in ("train", "val", "test")}
        regime = pd.Series(z["regime_vals"].astype(int), index=pd.DatetimeIndex(z["regime_dates"]))
        val_start, train_end = z["bounds"]
        return {"arrays": arrays, "tickers": tickers, "splits": splits, "regime": regime,
                "train_end": train_end, "val_start": val_start,
                "seq_len": int(z["seq_len"]), "horizon": int(z["horizon"])}


class FusionDataset(Dataset):
    def __init__(self, panel, split):
        self.arrays = panel["arrays"]
        self.samples = panel["splits"][split]
        self.seq_len = panel["seq_len"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ti, j = self.samples[i]
        a = self.arrays[ti]
        sl = slice(j - self.seq_len + 1, j + 1)
        return (
            torch.from_numpy(a["mom"][sl]),
            torch.from_numpy(a["text"][sl]),
            torch.from_numpy(a["text_mask"][sl]),
            torch.tensor(a["past_vol"][j], dtype=torch.float32),
            torch.tensor(np.log(max(a["target_vol"][j], 1e-6)), dtype=torch.float32),
        )


def sample_frame(panel, split):
    rows = []
    for ti, j in panel["splits"][split]:
        a = panel["arrays"][ti]
        sl = slice(j - panel["seq_len"] + 1, j + 1)
        rows.append({"ticker": panel["tickers"][ti], "date": a["dates"][j],
                     "target_end": a["target_end"][j], "target_vol": a["target_vol"][j],
                     "past_vol": a["past_vol"][j], "has_news": bool(a["text_mask"][sl].any()),
                     "news_today": bool(a["text_mask"][j])})
    df = pd.DataFrame(rows)
    df["regime"] = df["date"].map(panel["regime"]).fillna(0).astype(int)
    return df
