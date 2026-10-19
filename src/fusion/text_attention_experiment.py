"""GPU-only, out-of-sample headline-attention experiment against price and metadata controls."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from src.backtest.visualization import BG, PANEL, GOLD, GREEN, CYAN, WHITE, PURPLE

from src.fusion.event_experiment import (_portfolio, _portfolio_metrics, _prediction_frame,
                                         _scores, _signal_metrics, _train)
from src.fusion.event_model import PriceOnly, load_article_embeddings, load_event_panel
from src.fusion.headline_features import FEATURE_NAMES
from src.fusion.text_attention import MetadataDataset, TextAttention, TextEventDataset, load_headline_events


def run_experiment(panel_path, news_path, out_dir, cache_dir, cost_bps=10):
    """Fit real models with early stopping on a Colab GPU and save held-out comparisons."""
    try:
        from google.colab import drive  # noqa: F401 — local GPU is not a permitted runtime.
    except ImportError as exc:
        raise RuntimeError("Execute exclusivamente no Colab com GPU") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Ative uma GPU no Colab; treino/forward local são proibidos")
    if cost_bps < 0 or cost_bps >= 10_000:
        raise ValueError("Custo de transação em bps inválido")
    torch.manual_seed(42)
    np.random.seed(42)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = load_event_panel(panel_path)
    cached, cache_path = load_article_embeddings(news_path, cache_dir)
    events = load_headline_events(news_path, cached)
    train, val, test = (TextEventDataset(panel, events, split) for split in ("train", "val", "test"))
    meta_train, meta_val, meta_test = map(MetadataDataset, (train, val, test))
    device = torch.device("cuda")
    moment_dim = panel["arrays"][0]["mom"].shape[1]
    price = PriceOnly(moment_dim).to(device)
    price_history = _train(price, train, val, device)
    torch.save(price.state_dict(), out / "price_only.pt")
    text = TextAttention(moment_dim, test.dim).to(device)
    text_history = _train(text, train, val, device, residual=True, epochs=35)
    torch.save(text.state_dict(), out / "text_attention.pt")
    meta = TextAttention(moment_dim, 1).to(device)
    meta_history = _train(meta, meta_train, meta_val, device, residual=True, epochs=35)
    torch.save(meta.state_dict(), out / "metadata_only.pt")

    price_scores, mask = _scores(price, test, device)
    text_scores, text_mask = _scores(text, test, device, residual=True)
    meta_scores, meta_mask = _scores(meta, meta_test, device, residual=True)
    if not np.array_equal(mask, text_mask) or not np.array_equal(mask, meta_mask):
        raise ValueError("Máscaras de avaliação diferentes entre controles")
    values = {"Zero excesso": np.zeros_like(price_scores), "Texto + atenção": text_scores,
              "Metadados (contagem/idade)": meta_scores, "Somente preços": price_scores}
    predictions = {name: _prediction_frame(panel, test.samples, scores, mask)
                   for name, scores in values.items()}
    signal = pd.DataFrame({name: _signal_metrics(frame) for name, frame in predictions.items()}).T
    curves = {name: _portfolio(panel, frame, cost_bps) for name, frame in predictions.items()}
    first = next(iter(curves.values()))
    if any(not first.index.equals(curve.index) for curve in curves.values()):
        raise ValueError("Datas de carteira diferentes entre métodos")
    bh = first.copy()
    bh["strategy"] = bh["bh"]
    bh["turnover"] = bh["bh_turnover"]
    bh["skipped_rebalance"] = False
    curves["Comprar e manter"] = bh
    portfolio = pd.DataFrame({name: _portfolio_metrics(curve, cost_bps)
                              for name, curve in curves.items()}).T
    equity = pd.DataFrame({name: curve["bh"] if name == "Comprar e manter" else curve["strategy"]
                           for name, curve in curves.items()})
    months = pd.period_range(min(a["dates"][0] for a in panel["arrays"]),
                             max(a["dates"][-1] for a in panel["arrays"]), freq="M")
    coverage = (pd.DataFrame({"ticker": events["ticker"],
                              "month": pd.to_datetime(events["decision_date"]).to_period("M")})
                .groupby(["ticker", "month"]).size().rename("events")
                .reindex(pd.MultiIndex.from_product([panel["tickers"], months],
                                                    names=["ticker", "month"]), fill_value=0)
                .reset_index())
    coverage["month"] = coverage["month"].astype(str)
    split_coverage = pd.DataFrame([{"split": split, "samples": len(ds),
                                    "with_event": sum(bool(ds._news(int(ti), int(j))[1].any())
                                                      for ti, j in ds.samples)}
                                   for split, ds in (("train", train), ("val", val), ("test", test))])
    split_coverage["fraction_with_event"] = split_coverage["with_event"] / split_coverage["samples"]
    for name, frame in predictions.items():
        frame.to_csv(out / {"Zero excesso": "predictions_zero.csv", "Texto + atenção": "predictions_text.csv",
                            "Metadados (contagem/idade)": "predictions_metadata.csv",
                            "Somente preços": "predictions_price.csv"}[name], index=False)
    signal.to_csv(out / "signal_metrics.csv")
    portfolio.to_csv(out / "portfolio_metrics.csv")
    equity.to_csv(out / "equity_curves.csv")
    coverage.to_csv(out / "event_coverage_by_ticker_month.csv", index=False)
    split_coverage.to_csv(out / "split_coverage.csv", index=False)
    for name, history in (("price", price_history), ("text", text_history), ("metadata", meta_history)):
        history.to_csv(out / f"{name}_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for name, color in (("Zero excesso", WHITE), ("Texto + atenção", GOLD),
                        ("Metadados (contagem/idade)", PURPLE), ("Somente preços", CYAN),
                        ("Comprar e manter", GREEN)):
        ax.plot(equity.index, equity[name], color=color, linewidth=2,
                linestyle="--" if name == "Comprar e manter" else "-", label=name)
    ax.axhline(1, color=WHITE, alpha=0.2, linestyle="--")
    ax.set_title("Manchetes verificadas · retorno líquido no teste", color="white", fontsize=16)
    ax.set_ylabel("Capital (base 1; custos incluídos)", color="white", fontsize=14)
    ax.tick_params(colors="white", labelsize=12)
    ax.grid(alpha=0.12)
    for spine in ax.spines.values():
        spine.set_color(WHITE)
    ax.legend(fontsize=14, labelcolor="white", facecolor=PANEL, edgecolor=WHITE)
    fig.tight_layout()
    fig.savefig(out / "text_attention_vs_controls.png", dpi=300, facecolor=BG, edgecolor="none")
    plt.close(fig)
    (out / "results.md").write_text(
        "# Atenção ao texto de manchetes verificadas — avaliação fora da amostra\n\n"
        "Alvo: retorno simples futuro de cinco pregões menos retorno médio do universo no mesmo período. "
        "Manchetes auditadas, sem corpos de artigos. MiniLM congelado; características interpretáveis: "
        + ", ".join(FEATURE_NAMES) + ". A novidade compara até dez eventos anteriores de mesmo ticker, "
        "exclusivamente com datas de disponibilidade anteriores (não compara manchetes do mesmo dia). "
        "Eventos: últimas cinco sessões negociadas, até oito manchetes; a data de disponibilidade "
        "já atrasa a publicação um pregão. A rede aprende autoatenção entre eventos; MOMENT, "
        "volatilidade e retorno conhecido influenciam apenas a consulta de atenção cruzada, "
        "não a cabeça de previsão. Sem notícia recente, previsão textual e de controle é zero. "
        "O controle recebe exclusivamente contagem/máscara e idade, sem embeddings, novidade ou "
        "características textuais. Treino/validação/teste seguem separação temporal do painel, "
        "com rótulos purgados nas fronteiras; parada antecipada pela MSE da validação "
        "(apenas linhas com notícia para ambos os modelos de atenção).\n\n"
        "## Limitações\n\n" + panel["source_note"] + "\n\n"
        "O período de teste 2022-09 a 2023-11 já foi inspecionado na experiência anterior: "
        "esta comparação é diagnóstica, não uma confirmação independente. Sem notícia, "
        "previsão zero é apenas hipótese operacional; ausência de evento não comprova ausência "
        "de notícia. Janeiro de 2024 não tem cobertura auditável; não há avaliação de 2024 em diante. "
        "Universo ex post de ativos (incluindo ETFs) pode ter viés de seleção/sobrevivência. "
        "Horizontes de cinco pregões consecutivos se sobrepõem; correlações e diferenças "
        "de carteiras não estabelecem causalidade ou alpha futuro.\n\n"
        "## Execução e custos\n\n" + f"Cache: `{cache_path.name}`. Custo: {cost_bps} bps por valor "
        "negociado (compras e vendas). Carteiras positivas totalmente investidas rebalanceadas "
        "a cada cinco pregões após o fechamento; faltando previsão de algum ativo em data "
        "programada, mantêm os pesos. Comprar e manter adquire pesos iguais na mesma data "
        "inicial e não rebalanceia. Todas usam os mesmos pregões, retornos observados e custo de "
        "entrada; alpha anualizado é intercepto de retornos diários contra comprar e manter, "
        "beta é covariância/variância. MSE/IC também abrangem amostras sem manchetes; "
        "`mse_com_evento` mede exclusivamente linhas com manchetes recentes.\n\n"
        "## Cobertura por split\n\n```\n" + split_coverage.to_string(index=False) + "\n```\n\n"
        "## Métricas de previsão\n\n```\n" + signal.to_string() + "\n```\n\n"
        "## Carteira líquida\n\n```\n" + portfolio.to_string() + "\n```\n\n"
        "Curvas, previsões, históricos de treino, checkpoints e cobertura por ticker/mês: "
        "arquivos anexos nesta pasta.\n", encoding="utf-8")
    return {"signal": signal, "portfolio": portfolio, "coverage": coverage,
            "split_coverage": split_coverage, "equity": equity, "histories": {
                "price": price_history, "text": text_history, "metadata": meta_history},
            "cache": cache_path, "output": out}
