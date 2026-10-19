import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))

from src.fusion.dataset import load_panel, load_panel_file, FusionDataset, sample_frame
from src.fusion.models import build_model
from src.fusion.train import train_model
from src.fusion.evaluate import (
    predict, regression_metrics, backtest_metrics, per_regime_metrics,
    causal_text_effect, lead_lag_analysis,
)

BG = "#0d0d1a"
COLORS = {"ts_only": "#42A5F5", "text_only": "#CE93D8", "fusion": "#D4A843", "bh": "#AAAAAA"}
MODEL_NAMES = ["ts_only", "text_only", "fusion"]
METRIC_KEYS = ["rmse", "mae", "sign_acc", "auc", "ic", "total_ret_pct", "bh_ret_pct",
               "excess_ret_pct", "sharpe", "bh_sharpe", "max_dd_pct", "win_rate",
               "alpha_ann", "beta"]


def plot_equity(dailies, out_path):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    ax.set_facecolor(BG)
    for name, daily in dailies.items():
        eq = (1 + daily["strat"]).cumprod()
        ax.plot(eq.index, eq.values, label=name, color=COLORS.get(name, "#18FFFF"), linewidth=1.6)
    bh = (1 + list(dailies.values())[0]["bh"]).cumprod()
    ax.plot(bh.index, bh.values, label="buy & hold", color=COLORS["bh"], linewidth=1.4, linestyle="--")
    ax.set_title("Equity Curve — Teste (fora da amostra)", color="white", fontsize=16)
    ax.tick_params(colors="white", labelsize=12)
    ax.grid(alpha=0.12)
    ax.legend(fontsize=14, facecolor=BG, labelcolor="white", edgecolor="#333333")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor=BG, edgecolor="none")
    return fig


def plot_lead_lag(ll_df, out_path):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.bar(ll_df["lag"], ll_df["mean"], yerr=ll_df["std"] / np.sqrt(ll_df["count"]),
           color="#18FFFF", alpha=0.85, capsize=4, error_kw={"ecolor": "#AAAAAA"})
    ax.axhline(0, color="white", alpha=0.2, linestyle="--")
    ax.set_title("Lead-Lag: drift textual × retorno futuro (lag em dias)", color="white", fontsize=16)
    ax.set_xlabel("lag (dias)", color="white", fontsize=14)
    ax.set_ylabel("correlação média", color="white", fontsize=14)
    ax.tick_params(colors="white", labelsize=12)
    ax.grid(alpha=0.12)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor=BG, edgecolor="none")
    return fig


def run_ablation(panel, device, out_dir=None):
    device = torch.device(device)
    test_frame = sample_frame(panel, "test")
    results, dailies, attn_summary, hist_rows = {}, {}, {}, []

    for name in MODEL_NAMES:
        train_ds = FusionDataset(panel, "train")
        val_ds = FusionDataset(panel, "val")
        test_ds = FusionDataset(panel, "test")
        model = build_model(name)
        model, history = train_model(model, train_ds, val_ds, device)
        hist_rows.append({
            "model": name,
            "epochs": len(history),
            "train_mse": history[-1]["train_mse"],
            "val_mse": history[-1]["val_mse"],
        })

        preds, ent_ts, ent_txt = predict(model, test_ds, device, return_attn=(name == "fusion"))
        df = test_frame.copy()
        df["pred"] = preds
        df["ret_strat"] = np.sign(df["pred"]) * df["r_fwd"]

        reg = regression_metrics(df["r_fwd"].values, df["pred"].values)
        bt, daily = backtest_metrics(df)
        results[name] = {**reg, **bt}
        dailies[name] = daily

        pr = per_regime_metrics(df)
        results[name]["regime_0"] = pr.get(0, {})
        results[name]["regime_1"] = pr.get(1, {})

        if name == "fusion":
            if out_dir is not None:
                torch.save(model.state_dict(), Path(out_dir) / "fusion_cross_attention.pt")
            regime_vals = df["regime"].values
            attn_summary["entropy_ts_to_news"] = {
                "global": float(np.mean(ent_ts)),
                "regime_0": float(np.mean(ent_ts[regime_vals == 0])),
                "regime_1": float(np.mean(ent_ts[regime_vals == 1])),
            }
            attn_summary["entropy_news_to_ts"] = {
                "global": float(np.mean(ent_txt)),
                "regime_0": float(np.mean(ent_txt[regime_vals == 0])),
                "regime_1": float(np.mean(ent_txt[regime_vals == 1])),
            }
            attn_summary["causal_text"] = causal_text_effect(model, test_ds, device)

    rows = []
    for name, m in results.items():
        row = {"model": name}
        row.update({k: m.get(k) for k in METRIC_KEYS})
        row["ic_regime_0"] = m["regime_0"].get("ic")
        row["ic_regime_1"] = m["regime_1"].get("ic")
        row["sign_acc_regime_0"] = m["regime_0"].get("sign_acc")
        row["sign_acc_regime_1"] = m["regime_1"].get("sign_acc")
        rows.append(row)

    return {
        "table": pd.DataFrame(rows),
        "history": pd.DataFrame(hist_rows),
        "attn_summary": attn_summary,
        "lead_lag": lead_lag_analysis(panel),
        "dailies": dailies,
    }


def write_report(panel, result, out_dir):
    n_train = len(panel["splits"]["train"])
    n_val = len(panel["splits"]["val"])
    n_test = len(panel["splits"]["test"])
    lines = ["# Testes de Fusão Texto × Série Temporal (Deep Learning)\n"]
    lines.append(f"- Painel: {len(panel['tickers'])} tickers, embeddings texto (384d, lag 1 dia) + MOMENT (1024d)")
    lines.append(f"- Split temporal: treino até {pd.Timestamp(panel['val_start']).date()}, val até {pd.Timestamp(panel['train_end']).date()}, teste depois")
    lines.append(f"- Amostras treino/val/teste: {n_train}/{n_val}/{n_test}")
    lines.append("- Alvo: retorno forward 1 dia (`r_fwd`); estratégia: `sign(pred)` equal-weight vs buy & hold\n")
    lines.append("## Ablação (paper §2.1): unimodal vs multimodal\n")
    lines.append(result["table"].to_markdown(index=False))
    lines.append("\n## Entropia de atenção por regime (paper §3.1)\n")
    lines.append("```json")
    lines.append(json.dumps(result["attn_summary"].get("entropy_ts_to_news", {}), indent=2))
    lines.append(json.dumps(result["attn_summary"].get("entropy_news_to_ts", {}), indent=2))
    lines.append("```")
    lines.append("\n## Efeito causal do texto (paper §3.3, shuffle)\n")
    lines.append("```json")
    lines.append(json.dumps(result["attn_summary"].get("causal_text", {}), indent=2))
    lines.append("```")
    lines.append("\n## Lead-lag texto × preço (paper §2.3)\n")
    lines.append(result["lead_lag"].to_markdown(index=False))
    (Path(out_dir) / "results.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default=None, help="caminho para fusion_panel.npz (pula load_panel)")
    args = parser.parse_args()

    out_dir = _root / "outputs" / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    panel = load_panel_file(args.panel) if args.panel else load_panel()
    print(f"device: {device} | tickers: {len(panel['tickers'])} | "
          f"train/val/test: {len(panel['splits']['train'])}/{len(panel['splits']['val'])}/{len(panel['splits']['test'])}")

    result = run_ablation(panel, device, out_dir)

    for fig_fn, name in ((plot_equity, "equity_curves.png"), (plot_lead_lag, "lead_lag.png")):
        fig = fig_fn(result["dailies"] if fig_fn is plot_equity else result["lead_lag"], out_dir / name)
        plt.close(fig)

    result["table"].to_csv(out_dir / "results.csv", index=False)
    result["lead_lag"].to_csv(out_dir / "lead_lag.csv", index=False)
    write_report(panel, result, out_dir)

    print(result["table"].to_string(index=False))
    print(f"\nsalvo em {out_dir}")


if __name__ == "__main__":
    main()
