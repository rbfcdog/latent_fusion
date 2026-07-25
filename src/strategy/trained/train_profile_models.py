#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_root))

from src.strategy.trained.profile_models import TrainedProfileStrategy

CHECKPOINT_DIR = _root / "src/strategy/trained/checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

PROFILES = ["conservador", "moderado", "arrojado"]
MODEL_TYPES = ["ridge", "elasticnet", "mlp"]

def load_data():
    prices = pd.read_parquet(_root / "data/lse_market_data/combined_1d.parquet")
    prices["timestamp"] = pd.to_datetime(prices["timestamp"]).dt.tz_localize(None).dt.normalize()
    pivot = prices.pivot_table(index="timestamp", columns="symbol", values="close")
    pivot = pivot.ffill().bfill()
    returns = pivot.pct_change().fillna(0)

    meta_sp = pd.read_csv(_root / "cache/text/top50_daily_metadata.csv")
    meta_sp["date"] = pd.to_datetime(meta_sp["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    meta_sp = meta_sp.dropna(subset=["date"])
    emb_all = np.load(_root / "cache/text/top50_daily_embeddings.npy")

    nasdaq_tickers = set(prices[prices["asset_group"] == "nasdaq"]["symbol"].unique())
    sp_tickers = sorted(nasdaq_tickers & set(meta_sp["ticker"].unique()))
    if not sp_tickers:
        sp_tickers = sorted(meta_sp["ticker"].unique())[:10]

    sp_meta = meta_sp[meta_sp["ticker"].isin(sp_tickers)]
    sp_meta = sp_meta[sp_meta["date"] >= "2020-01-01"]

    text_rows = []
    for _, row in sp_meta.iterrows():
        text_rows.append({"date": row["date"], "ticker": row["ticker"], "emb": emb_all[row.name]})

    text_df = pd.DataFrame(text_rows)
    text_df["date"] = pd.to_datetime(text_df["date"]).dt.normalize()

    from sklearn.decomposition import PCA
    all_embs = np.vstack([r["emb"] for r in text_rows])
    pca = PCA(n_components=32, random_state=42)
    all_embs_pca = pca.fit_transform(all_embs)
    emb_cols = [f"emb_{i}" for i in range(32)]
    text_df[emb_cols] = all_embs_pca

    text_pivot = text_df.pivot_table(index="date", columns="ticker", values=emb_cols)
    text_pivot.columns = [f"{t}_{c}" for c, t in text_pivot.columns]

    common_tickers = sorted(set(sp_tickers) & set(returns.columns))
    if len(common_tickers) < 3:
        return None, None, None

    X, y = [], []
    for t in common_tickers:
        ticker_cols = [c for c in text_pivot.columns if c.startswith(f"{t}_emb_")]
        if not ticker_cols:
            continue
        ticker_emb = text_pivot[ticker_cols].reindex(returns.index)
        ticker_ret = returns[t].shift(-1)
        aligned = pd.concat([ticker_emb, ticker_ret.rename("target")], axis=1).dropna()
        if len(aligned) < 50:
            continue
        X.append(aligned[ticker_cols].values)
        y.append(aligned["target"].values)

    if not X:
        return None, None, None

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y, common_tickers

def main():
    print("Carregando dados...")
    t0 = time.time()
    X, y, tickers = load_data()
    if X is None or y is None:
        print("ERRO: nao foi possivel carregar dados de treinamento")
        print("Certifique-se de que cache/text/top50_daily_embeddings.npy existe")
        return 1
    print(f"Dados: X={X.shape} y={y.shape} tickers={len(tickers)} ({time.time()-t0:.1f}s)")

    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"Split: train={len(X_train)} test={len(X_test)}")

    results = {}

    for profile_name in PROFILES:
        for model_type in MODEL_TYPES:
            key = f"{profile_name}/{model_type}"
            print(f"\nTreinando {key}...")
            t1 = time.time()

            strategy = TrainedProfileStrategy(profile_name=profile_name)
            try:
                strategy.fit(X_train, y_train, model_type=model_type, pca_dim=32)
            except Exception as e:
                print(f"  ERRO: {e}")
                results[key] = {"error": str(e)}
                continue

            preds_train = strategy.predict(X_train)
            preds_test = strategy.predict(X_test)

            train_corr = np.corrcoef(preds_train, y_train)[0, 1] if len(preds_train) > 1 else 0
            test_corr = np.corrcoef(preds_test, y_test)[0, 1] if len(preds_test) > 1 else 0

            train_mse = np.mean((preds_train - y_train) ** 2)
            test_mse = np.mean((preds_test - y_test) ** 2)

            pred_vol = np.std(preds_test)
            signal_strength = np.mean(np.abs(preds_test))
            nonzero_pct = (np.abs(preds_test) > 0.0001).mean() * 100

            ckpt_path = CHECKPOINT_DIR / f"{profile_name}_{model_type}.pkl"
            strategy.save(str(ckpt_path))

            elapsed = time.time() - t1
            info = {
                "profile": profile_name,
                "model": model_type,
                "train_corr": round(float(train_corr), 4),
                "test_corr": round(float(test_corr), 4),
                "train_mse": round(float(train_mse), 6),
                "test_mse": round(float(test_mse), 6),
                "pred_vol": round(float(pred_vol), 6),
                "signal_strength": round(float(signal_strength), 6),
                "nonzero_pct": round(float(nonzero_pct), 1),
                "train_s": round(elapsed, 1),
                "checkpoint": str(ckpt_path.name),
            }
            results[key] = info
            print(f"  corr={test_corr:+.4f} mse={test_mse:.6f} vol={pred_vol:.6f} "
                  f"strength={signal_strength:.6f} nonzero={nonzero_pct:.1f}% ({elapsed:.1f}s)")

    print("\n" + "=" * 70)
    print("RESUMO — Modelos Treinados por Perfil")
    print("=" * 70)
    for k, v in sorted(results.items()):
        if "error" in v:
            print(f"  {k:<30s} ERRO: {v['error']}")
        else:
            print(f"  {k:<30s} corr={v['test_corr']:+.4f} vol={v['pred_vol']:.6f} "
                  f"strength={v['signal_strength']:.6f} nonzero={v['nonzero_pct']:.1f}%")

    results_path = CHECKPOINT_DIR / "profile_training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em {results_path}")
    print(f"Checkpoints salvos em {CHECKPOINT_DIR}")

if __name__ == "__main__":
    sys.exit(main() or 0)
