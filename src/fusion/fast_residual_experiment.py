from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.backtest.visualization import BG, PANEL, GOLD, CYAN, GREEN, PURPLE, WHITE
from src.fusion.event_experiment import _scores, _train
from src.fusion.event_model import EventResidual, load_event_panel
from src.fusion.fast_demo_experiment import _predict, _summarize
from src.fusion.models import TSOnlyPredictor
from src.fusion.text_attention import MetadataDataset, TextEventDataset


class _SavedNumericBaseline(nn.Module):

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, mom, past_vol, known_ret):
        numeric = self.predictor.numeric(mom)
        score = self.predictor.head(torch.cat((numeric, torch.log(past_vol.clamp_min(1e-6))[:, None]), dim=1))
        return score, numeric


def load_saved_demo(source):
    source = Path(source)
    panel = load_event_panel(source / "fast_demo_panel.npz")
    with np.load(source / "fast_demo_events.npz", allow_pickle=False) as saved:
        events = {key: saved[key] for key in
                  ("ticker", "decision_date", "published_at", "embedding", "features", "novelty")}
    provenance = json.loads((source / "fast_demo_provenance.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(source / "ts_only.pt", map_location="cpu", weights_only=True)
    with np.load(source / "fast_demo_panel.npz", allow_pickle=False) as saved:
        created_at = str(saved["created_at_utc"])
    if (checkpoint["model"] != "ts_only" or checkpoint["provenance"] != provenance or
            created_at != provenance["created_at_utc"] or
            panel["tickers"] != provenance["assets"] or
            len(events["embedding"]) != provenance["article_embeddings"] or
            checkpoint["d_ts"] != panel["arrays"][0]["mom"].shape[1] or
            checkpoint["d_text"] != events["embedding"].shape[1] + events["features"].shape[1] + 2 or
            any(a["mom"].shape[1] != checkpoint["d_ts"] for a in panel["arrays"]) or
            any(len(events[key]) != len(events["embedding"]) for key in events) or
            not np.isfinite(events["embedding"]).all() or
            not np.isfinite(events["features"]).all() or
            not np.isfinite(events["novelty"]).all()):
        raise ValueError("Embeddings e checkpoint numérico não pertencem à mesma execução")
    return panel, events, checkpoint, provenance


def _plot(equity, path):
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for name, label, color in (("ts_only", "Somente série temporal", GOLD),
                                ("metadata", "Série + idade/contagem", PURPLE),
                                ("text_residual", "Série + atenção às manchetes", CYAN),
                                ("buy_hold", "Comprar e manter (8 ativos)", GREEN)):
        ax.plot(equity.index, equity[name], label=label, color=color,
                linewidth=2.2, linestyle="--" if name == "buy_hold" else "-")
    ax.axhline(1, color=WHITE, alpha=0.2, linestyle="--")
    ax.set_title("Atenção residual · retorno líquido no teste", fontsize=16, color="white")
    ax.set_ylabel("Capital (base 1; após custos)", fontsize=14, color="white")
    ax.tick_params(colors="white", labelsize=12)
    ax.grid(alpha=0.12)
    ax.legend(facecolor=PANEL, labelcolor="white", edgecolor=WHITE, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=BG, edgecolor="none")
    plt.close(fig)


def run_experiment(saved_dir, out_dir, *, cost_bps=10):
    try:
        import google.colab
    except ImportError as exc:
        raise RuntimeError("Execute exclusivamente no Google Colab com GPU") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Ative uma GPU no Colab; nenhum forward local é permitido")
    if not 0 <= cost_bps < 10_000:
        raise ValueError("Custo de transação em bps inválido")
    panel, events, checkpoint, provenance = load_saved_demo(saved_dir)
    text = {split: TextEventDataset(panel, events, split) for split in ("train", "val", "test")}
    baseline = TSOnlyPredictor(d_ts=checkpoint["d_ts"]).to("cuda")
    baseline.load_state_dict(checkpoint["state_dict"], strict=True)
    baseline.eval()
    baseline_scores, mask = _predict(baseline, text["test"])
    scores = {"ts_only": baseline_scores}
    histories = {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for name, datasets, dimension in (
        ("metadata", {split: MetadataDataset(ds) for split, ds in text.items()}, 1),
        ("text_residual", text, text["train"].dim),
    ):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = EventResidual(_SavedNumericBaseline(baseline), event_dim=dimension,
                              max_residual=0.02).to("cuda")
        nn.init.zeros_(model.residual[-1].weight)
        nn.init.zeros_(model.residual[-1].bias)
        histories[name] = _train(model, datasets["train"], datasets["val"],
                                  torch.device("cuda"), residual=True, epochs=8, patience=2)
        scores[name], current_mask = _scores(model, datasets["test"], torch.device("cuda"), residual=True)
        if not np.array_equal(current_mask, mask):
            raise ValueError("As comparações devem usar as mesmas manchetes e datas")
        torch.save({"model": name, "state_dict": model.state_dict(),
                    "source_provenance": provenance, "baseline_seed": checkpoint["seed"]}, out / f"{name}.pt")
    scores["zero"] = np.zeros_like(baseline_scores)
    predictions, signal, portfolio, equity = _summarize(panel, text["test"].samples, mask, scores, cost_bps)
    for name, frame in predictions.items():
        frame.to_csv(out / f"predictions_{name}.csv", index=False)
    for name, history in histories.items():
        history.to_csv(out / f"{name}_history.csv", index=False)
    signal.to_csv(out / "signal_metrics.csv")
    portfolio.to_csv(out / "portfolio_metrics.csv")
    equity.to_csv(out / "equity_curves.csv")
    (out / "run_config.json").write_text(json.dumps({
        "embedding_provenance": provenance, "baseline_seed": checkpoint["seed"],
        "residual_seed": 42, "max_epochs": 8, "patience": 2,
        "max_residual": 0.02, "cost_bps": cost_bps,
        "models": ["ts_only", "metadata", "text_residual", "zero", "buy_hold"],
        "source_note": panel["source_note"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(equity, out / "residual_vs_bh.png")
    (out / "results.md").write_text(
        "# Atenção residual — demonstração exploratória\n\n"
        "MOMENT-1-small e MiniLM são os embeddings já gerados e auditados pela fusão rápida; "
        "este experimento não executa os encoders de novo. O mesmo checkpoint numérico fica "
        "congelado nos três modelos. O residual começa exatamente em zero, consulta por atenção "
        "as manchetes ou apenas idade/contagem, e retorna à previsão numérica sem notícias. "
        "Treino e escolha da época são feitos apenas em treino/validação; o residual usa MSE das "
        "linhas com notícia na validação. Mesmo universo, horizonte, datas e custo de "
        f"{cost_bps} bps para as três carteiras. Previsão zero é referência de erro, não carteira.\n\n"
        "O teste 2022–23 já foi inspecionado: qualquer melhora é exploratória, não uma nova "
        "confirmação; universo selecionado ex post e feed indisponível após novembro de 2023.\n\n"
        "## MSE na validação (linhas com notícias)\n\n```\n" +
        pd.DataFrame({name: history.loc[history["val_mse"].idxmin(), ["epoch", "val_mse"]]
                      for name, history in histories.items()}).T.to_string() + "\n```\n\n"
        "## Erro e correlação de sinal no teste\n\n```\n" + signal.to_string() + "\n```\n\n"
        "## Carteiras após custos\n\n```\n" + portfolio.to_string() + "\n```\n\n",
        encoding="utf-8")
    return {"signal": signal, "portfolio": portfolio, "equity": equity,
            "histories": histories, "output": out, "provenance": provenance}
