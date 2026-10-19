"""Fresh-embedding, Colab-only comparison of numeric and headline-attention signals."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.backtest.visualization import BG, PANEL, GOLD, CYAN, GREEN, WHITE
from src.fusion.event_experiment import _portfolio, _portfolio_metrics, _prediction_frame, _signal_metrics
from src.fusion.fast_demo_data import prepare_demo_inputs
from src.fusion.models import FusionPredictor, TSOnlyPredictor
from src.fusion.text_attention import TextEventDataset


def _fit(model, train, val, *, epochs=8, patience=2):
    train_loader = DataLoader(train, batch_size=256, shuffle=True, pin_memory=True,
                              generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val, batch_size=512, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_loss, best_weights, stale, rows = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            mom, vol, _, text, mask, target = (item.to("cuda", non_blocking=True) for item in batch)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(mom, text, mask, vol)[0][:, 0]
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(target)
            count += len(target)
        model.eval()
        val_total, val_count = 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                mom, vol, _, text, mask, target = (item.to("cuda", non_blocking=True) for item in batch)
                prediction = model(mom, text, mask, vol)[0][:, 0]
                val_total += torch.nn.functional.mse_loss(prediction, target, reduction="sum").item()
                val_count += len(target)
        if not count or not val_count:
            raise ValueError("Treino/validação sem amostras")
        value = val_total / val_count
        rows.append({"epoch": epoch + 1, "train_mse": total / count, "val_mse": value})
        if value < best_loss - 1e-8:
            best_loss, best_weights, stale = value, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_weights)
    return pd.DataFrame(rows)


def _predict(model, dataset):
    model.eval()
    scores, masks = [], []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=512, pin_memory=True):
            mom, vol, _, text, mask, _ = (item.to("cuda", non_blocking=True) for item in batch)
            scores.append(model(mom, text, mask, vol)[0][:, 0].cpu().numpy())
            masks.append(mask.any(dim=1).cpu().numpy())
    return np.concatenate(scores), np.concatenate(masks)


def _summarize(panel, samples, mask, scores, cost_bps):
    """Pure data summary: identical held-out rows and calendar for every signal."""
    if any(len(values) != len(samples) for values in (*scores.values(), mask)):
        raise ValueError("Previsões e máscaras desalinhadas com o teste")
    predictions = {name: _prediction_frame(panel, samples, values, mask)
                   for name, values in scores.items()}
    first = next(iter(predictions.values()))
    for frame in predictions.values():
        if not frame[["ticker", "date"]].equals(first[["ticker", "date"]]):
            raise ValueError("Linhas de avaliação diferentes entre modelos")
    signal = pd.DataFrame({name: _signal_metrics(frame) for name, frame in predictions.items()}).T
    curves = {name: _portfolio(panel, frame, cost_bps) for name, frame in predictions.items()
              if name != "zero"}
    if not curves or any(not next(iter(curves.values())).index.equals(c.index) for c in curves.values()):
        raise ValueError("Datas de carteira diferentes entre modelos")
    bh = next(iter(curves.values())).copy()
    bh["strategy"] = bh["bh"]
    bh["turnover"] = bh["bh_turnover"]
    bh["skipped_rebalance"] = False
    curves["buy_hold"] = bh
    portfolio = pd.DataFrame({name: _portfolio_metrics(curve, cost_bps)
                              for name, curve in curves.items()}).T
    equity = pd.DataFrame({name: curve["strategy"] for name, curve in curves.items()})
    return predictions, signal, portfolio, equity


def _plot(equity, path):
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for name, label, color in (("ts_only", "Somente série temporal", GOLD),
                               ("fusion", "Série + manchetes (atenção)", CYAN),
                               ("buy_hold", "Comprar e manter (8 ativos)", GREEN)):
        ax.plot(equity.index, equity[name], label=label, color=color,
                linewidth=2.2, linestyle="--" if name == "buy_hold" else "-")
    ax.axhline(1, color=WHITE, alpha=0.2, linestyle="--")
    ax.set_title("Demonstração de fusão · retorno líquido no teste", fontsize=16, color="white")
    ax.set_ylabel("Capital (base 1; após custos)", fontsize=14, color="white")
    ax.tick_params(colors="white", labelsize=12)
    ax.grid(alpha=0.12)
    for spine in ax.spines.values():
        spine.set_color(WHITE)
    ax.legend(fontsize=14, labelcolor="white", facecolor=PANEL, edgecolor=WHITE)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=BG, edgecolor="none")
    plt.close(fig)


def run_experiment(panel_path, news_path, out_dir, *, cost_bps=10, window=64, batch_size=32):
    """Generate both embedding families anew and train both heads only in Colab CUDA."""
    try:
        import google.colab  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Execute exclusivamente no Google Colab com GPU") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Ative uma GPU no Colab; treino/forward local são proibidos")
    if not 0 <= cost_bps < 10_000:
        raise ValueError("Custo de transação em bps inválido")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel, events, provenance = prepare_demo_inputs(panel_path, news_path, out,
                                                      window=window, batch_size=batch_size)
    train, val, test = (TextEventDataset(panel, events, split) for split in ("train", "val", "test"))
    dimension = panel["arrays"][0]["mom"].shape[1]
    torch.manual_seed(42)
    baseline = TSOnlyPredictor(d_ts=dimension).to("cuda")
    torch.manual_seed(42)
    fusion = FusionPredictor(d_ts=dimension, d_text=test.dim).to("cuda")
    models = {"ts_only": baseline, "fusion": fusion}
    histories, scores, mask = {}, {}, None
    for name, model in models.items():
        histories[name] = _fit(model, train, val)
        torch.save({"model": name, "state_dict": model.state_dict(), "d_ts": dimension,
                    "d_text": test.dim, "seed": 42, "provenance": provenance}, out / f"{name}.pt")
        scores[name], current_mask = _predict(model, test)
        if mask is not None and not np.array_equal(mask, current_mask):
            raise ValueError("Máscaras distintas entre modelos")
        mask = current_mask
    scores["zero"] = np.zeros_like(scores["ts_only"])
    predictions, signal, portfolio, equity = _summarize(panel, test.samples, mask, scores, cost_bps)
    months = pd.period_range(min(a["dates"][0] for a in panel["arrays"]),
                             max(a["dates"][-1] for a in panel["arrays"]), freq="M")
    coverage = (pd.DataFrame({"ticker": events["ticker"],
                              "month": pd.to_datetime(events["decision_date"]).to_period("M")})
                .groupby(["ticker", "month"]).size().rename("titles")
                .reindex(pd.MultiIndex.from_product([panel["tickers"], months],
                                                    names=["ticker", "month"]), fill_value=0)
                .reset_index())
    coverage["month"] = coverage["month"].astype(str)
    split_coverage = pd.DataFrame([
        {"split": name, "samples": len(ds),
         "with_event": sum(bool(ds._news(int(ti), int(j))[1].any()) for ti, j in ds.samples)}
        for name, ds in (("train", train), ("val", val), ("test", test))])
    split_coverage["fraction_with_event"] = split_coverage["with_event"] / split_coverage["samples"]
    provenance_table = pd.DataFrame([{"source": key, "provenance": str(value)}
                                     for key, value in provenance.items()])
    for name, frame in predictions.items():
        frame.to_csv(out / f"predictions_{name}.csv", index=False)
    for name, history in histories.items():
        history.to_csv(out / f"{name}_history.csv", index=False)
    signal.to_csv(out / "signal_metrics.csv")
    portfolio.to_csv(out / "portfolio_metrics.csv")
    equity.to_csv(out / "equity_curves.csv")
    coverage.to_csv(out / "event_coverage_by_ticker_month.csv", index=False)
    split_coverage.to_csv(out / "split_coverage.csv", index=False)
    provenance_table.to_csv(out / "source_provenance.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({
        "models": {"ts_only": "TSOnlyPredictor", "fusion": "FusionPredictor"},
        "embedding_provenance": provenance, "seed": 42, "max_epochs": 8,
        "patience": 2, "window": window, "cost_bps": cost_bps,
        "tickers": list(panel["tickers"]), "source_note": panel["source_note"],
        "splits": split_coverage.to_dict(orient="records")}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    _plot(equity, out / "fast_fusion_vs_bh.png")
    (out / "results.md").write_text(
        "# Demonstração de fusão rápida — não é prova de alfa\n\n"
        "Embeddings MOMENT-1-small de retornos conhecidos e MiniLM de títulos verificados "
        "gerados nesta execução; não usar os vetores antigos do painel nem o cache textual. "
        "Corpos dos artigos não confiáveis foram descartados. As notícias já têm atraso de "
        "um pregão na data de decisão. Treino, validação e teste cronológicos com purga "
        "de rótulos, parada antecipada pela validação e as mesmas linhas (inclusive sem notícia) "
        "para ambos os modelos. A fusão retorna à cabeça numérica quando não há manchete "
        "recente. Alvo: retorno futuro de cinco pregões menos a média dos oito ativos.\n\n"
        "O teste 2022–23 já havia sido inspecionado: a comparação não é confirmação nova. "
        "Universo escolhido ex post; viés de seleção/sobrevivência, períodos sem cobertura "
        "e horizontes sobrepostos limitam a interpretação. Ausência de manchete não prova "
        "ausência de notícias. Não inferir causalidade nem desempenho futuro.\n\n"
        f"Cada carteira usa pesos positivos e os mesmos pregões; custo de {cost_bps} bps por valor "
        "negociado em compras/vendas (inclusive entrada). BH compra pesos iguais na primeira "
        "data e mantém as quantidades. Rebalanceamento a cada cinco pregões, após o fechamento. "
        "Alfa e beta derivam dos retornos diários da estratégia contra BH. "
        "A previsão zero é referência de erro, não carteira de investimento.\n\n"
        "## Cobertura por partição\n\n```\n" + split_coverage.to_string(index=False) + "\n```\n\n"
        "## Erro e correlação de sinal\n\n```\n" + signal.to_string() + "\n```\n\n"
        "## Carteiras após custos\n\n```\n" + portfolio.to_string() + "\n```\n\n"
        "Fonte: " + panel["source_note"] + "\n", encoding="utf-8")
    return {"signal": signal, "portfolio": portfolio, "equity": equity, "coverage": coverage,
            "split_coverage": split_coverage, "provenance": provenance_table,
            "histories": histories, "output": out}
