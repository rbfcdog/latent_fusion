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

from src.fusion.dataset import FusionDataset, load_panel, load_panel_file, sample_frame
from src.fusion.evaluate import grouped_metrics, predict, shuffled_text_effect, volatility_metrics
from src.fusion.models import build_model
from src.fusion.train import train_model

BG = "#0d0d1a"
COLORS = {"ts_only": "#42A5F5", "text_only": "#CE93D8", "fusion": "#D4A843",
          "persistence": "#FF9800", "observed": "#AAAAAA"}


def plot_volatility(dailies, out_path):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    ax.set_facecolor("#1a1a2e")
    observed = next(iter(dailies.values()))
    ax.plot(observed.index, observed["target_vol"], color=COLORS["observed"],
            linestyle="--", label="realizada", linewidth=1.4)
    ax.plot(observed.index, observed["past_vol"], color=COLORS["persistence"],
            label="persistência", linewidth=1.4)
    for name, daily in dailies.items():
        ax.plot(daily.index, daily["pred"], color=COLORS[name], label=name, linewidth=1.4)
    ax.set_title("Volatilidade futura de 20 pregões — teste", color="white", fontsize=16)
    ax.set_ylabel("volatilidade anualizada", color="white", fontsize=14)
    ax.tick_params(colors="white", labelsize=12)
    ax.grid(alpha=0.12)
    ax.legend(fontsize=14, facecolor=BG, labelcolor="white", edgecolor="#333333")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor=BG, edgecolor="none")
    return fig


def run_ablation(panel, device, out_dir=None):
    device = torch.device(device)
    test = FusionDataset(panel, "test")
    test_frame = sample_frame(panel, "test")
    targets = test_frame["target_vol"].to_numpy()
    results = {"persistence": volatility_metrics(targets, test_frame["past_vol"].to_numpy())}
    dailies, predictions, hist_rows, grouped_rows = {}, {}, [], []
    attn_summary = {}
    persistence = test_frame.copy()
    persistence["pred"] = persistence["past_vol"]
    for group_col in ("regime", "has_news"):
        for label, metrics in grouped_metrics(persistence, group_col).items():
            grouped_rows.append({"model": "persistence", "group": group_col, "value": label, **metrics})

    for name in ("ts_only", "text_only", "fusion"):
        torch.manual_seed(42)
        model = build_model(name, d_ts=panel["arrays"][0]["mom"].shape[1],
                            d_text=panel["arrays"][0]["text"].shape[1])
        model, history = train_model(model, FusionDataset(panel, "train"),
                                     FusionDataset(panel, "val"), device)
        hist_rows.append({"model": name, "epochs": len(history),
                          "train_log_mse": history[-1]["train_mse"],
                          "val_log_mse": history[-1]["val_mse"]})
        preds, entropy = predict(model, test, device, return_attn=name == "fusion")
        frame = test_frame.copy()
        frame["pred"] = preds
        predictions[name] = frame
        results[name] = volatility_metrics(targets, preds)
        dailies[name] = frame.groupby("date")[["target_vol", "past_vol", "pred"]].mean().sort_index()
        for group_col in ("regime", "has_news"):
            for label, metrics in grouped_metrics(frame, group_col).items():
                grouped_rows.append({"model": name, "group": group_col, "value": label, **metrics})

        if name == "fusion":
            if out_dir is not None:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), Path(out_dir) / "fusion_vol_20d_attention.pt")
            attn_summary = {
                "entropy_news_only": float(np.nanmean(entropy)),
                "entropy_by_regime": {
                    str(r): float(np.nanmean(entropy[frame["regime"].to_numpy() == r]))
                    for r in (0, 1)
                },
                "text_permutation": shuffled_text_effect(model, test, device),
            }

    table = pd.DataFrame([{"model": name, **metrics} for name, metrics in results.items()])
    groups = pd.DataFrame(grouped_rows)
    baseline = predictions["ts_only"]
    for group_col in ("regime", "has_news"):
        for label, part in baseline.groupby(group_col):
            numeric = volatility_metrics(part["target_vol"], part["pred"])["rmse"]
            match = (groups["group"] == group_col) & (groups["value"] == str(label))
            groups.loc[match, "gain_vs_ts_rmse"] = numeric - groups.loc[match, "rmse"]
    return {"table": table, "groups": groups, "history": pd.DataFrame(hist_rows),
            "attn_summary": attn_summary, "dailies": dailies}


def write_report(panel, result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result["table"].to_csv(out_dir / "results.csv", index=False)
    result["groups"].to_csv(out_dir / "groups.csv", index=False)
    result["history"].to_csv(out_dir / "history.csv", index=False)
    lines = ["# Fusão para volatilidade realizada futura (20 pregões)", "",
             f"Painel: {len(panel['tickers'])} tickers; alvo: raiz da média de retornos logarítmicos quadrados dos próximos {panel['horizon']} pregões × √252.",
             "Texto atrasado um pregão, sem imputação de notícias; MOMENT e vol passada conhecidos na decisão.",
             f"Treino até {pd.Timestamp(panel['val_start']).date()}, validação até {pd.Timestamp(panel['train_end']).date()}, teste depois. Rótulos de treino/validação purgados na fronteira.",
             "Métricas em volatilidade anualizada (0,20 = 20%); R² relativo à média do teste. Persistência = volatilidade realizada dos 20 pregões anteriores. Janelas futuras se sobrepõem; diferenças pequenas não implicam significância estatística.",
             "", "## Ablação", "", result["table"].to_html(index=False),
             "", "## Regime conhecido e presença de notícia", "", result["groups"].to_html(index=False),
             "", "## Atenção e permutação de texto (diagnóstico, não evidência causal)", "",
             "```json", json.dumps(result["attn_summary"], indent=2), "```"]
    (out_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", help="fusion_vol_panel.npz")
    args = parser.parse_args()
    out_dir = _root / "outputs" / "fusion_vol"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel_file(args.panel) if args.panel else load_panel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_ablation(panel, device, out_dir)
    fig = plot_volatility(result["dailies"], out_dir / "volatility.png")
    plt.close(fig)
    write_report(panel, result, out_dir)
    print(result["table"].to_string(index=False))


if __name__ == "__main__":
    main()
