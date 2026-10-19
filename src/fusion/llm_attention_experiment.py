"""Colab-only incremental test of issuer-grounded headline facts over MiniLM."""

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
from src.fusion.llm_attention import LLMEventDataset, align_llm_events
from src.fusion.llm_headlines import LLM_FEATURE_NAMES, MODEL_ID, MODEL_REVISION, extract_llm_features
from src.fusion.text_attention import TextAttention


def _audit(features, status):
    valid = status == "ok"
    invalid_count = int((~valid).sum())
    if not valid.any():
        raise ValueError("Nenhuma extração LLM válida")
    groups = {"papel_desconhecido": (0, 2), "classe_desconhecida": (2, 9),
              "direcao_desconhecida": (9, 11), "certeza_desconhecida": (11, 14),
              "sem_fato_estruturado": (0, len(LLM_FEATURE_NAMES))}
    return {"total": len(status), "valid": int(valid.sum()), "invalid": invalid_count,
            "invalid_pct": 100 * invalid_count / len(status), **{
                key + "_pct_entre_validos": 100 * float(np.mean(~features[valid, start:end].any(axis=1)))
                for key, (start, end) in groups.items()}}


def run_experiment(panel_path, news_path, out_dir, embedding_cache_dir, llm_cache_dir,
                   cost_bps=10, llm_result=None):
    """Only train after completed, provenance-aligned extraction on a Colab GPU."""
    try:
        from google.colab import drive  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Execute exclusivamente no Colab com GPU") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Ative uma GPU no Colab; treino/forward local são proibidos")
    if cost_bps < 0 or cost_bps >= 10_000:
        raise ValueError("Custo de transação em bps inválido")
    facts, facts_path = llm_result if llm_result is not None else extract_llm_features(news_path, llm_cache_dir)
    cached, embedding_path = load_article_embeddings(news_path, embedding_cache_dir)
    events = align_llm_events(news_path, cached, facts)
    audit = _audit(events["features"], events["status"])
    if audit["invalid_pct"] > 5:
        raise ValueError(f"Extrações LLM inválidas: {audit['invalid_pct']:.2f}% > 5%; revisar cache antes do treino")
    panel = load_event_panel(panel_path)
    train, val, test = (LLMEventDataset(panel, events, split) for split in ("train", "val", "test"))
    emb_train, emb_val, emb_test = (LLMEventDataset(panel, events, split, embeddings_only=True)
                                    for split in ("train", "val", "test"))
    for full, embedding in ((train, emb_train), (val, emb_val), (test, emb_test)):
        if not np.array_equal(full.samples, embedding.samples):
            raise ValueError("Splits diferentes entre modelos de manchete")
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda")
    moment_dim = panel["arrays"][0]["mom"].shape[1]
    price = PriceOnly(moment_dim).to(device)
    price_history = _train(price, train, val, device)
    full = TextAttention(moment_dim, test.dim).to(device)
    full_history = _train(full, train, val, device, residual=True, epochs=35)
    embedding = TextAttention(moment_dim, emb_test.dim).to(device)
    embedding_history = _train(embedding, emb_train, emb_val, device, residual=True, epochs=35)
    price_scores, mask = _scores(price, test, device)
    full_scores, full_mask = _scores(full, test, device, residual=True)
    embedding_scores, emb_mask = _scores(embedding, emb_test, device, residual=True)
    if not np.array_equal(mask, full_mask) or not np.array_equal(mask, emb_mask):
        raise ValueError("Máscaras de avaliação diferentes entre modelos")
    names = ("LLM + MiniLM", "Somente MiniLM", "Somente preços", "Zero excesso")
    scores = (full_scores, embedding_scores, price_scores, np.zeros_like(price_scores))
    predictions = {name: _prediction_frame(panel, test.samples, score, mask)
                   for name, score in zip(names, scores)}
    signal = pd.DataFrame({name: _signal_metrics(frame) for name, frame in predictions.items()}).T
    curves = {name: _portfolio(panel, frame, cost_bps) for name, frame in predictions.items()}
    reference = curves[names[0]]
    if any(not reference.index.equals(curve.index) for curve in curves.values()):
        raise ValueError("Datas de carteira diferentes entre métodos")
    bh = reference.copy()
    bh["strategy"] = bh["bh"]
    bh["turnover"] = bh["bh_turnover"]
    bh["skipped_rebalance"] = False
    curves["Comprar e manter"] = bh
    portfolio = pd.DataFrame({name: _portfolio_metrics(curve, cost_bps)
                              for name, curve in curves.items()}).T
    equity = pd.DataFrame({name: curve["strategy"] for name, curve in curves.items()})
    split_coverage = pd.DataFrame([{"split": split, "samples": len(dataset),
                                    "with_event": sum(bool(dataset._news(int(ti), int(j))[1].any())
                                                      for ti, j in dataset.samples)}
                                   for split, dataset in (("train", train), ("val", val), ("test", test))])
    split_coverage["fraction_with_event"] = split_coverage["with_event"] / split_coverage["samples"]
    coverage = (pd.DataFrame({"ticker": events["ticker"],
                              "month": pd.to_datetime(events["decision_date"]).to_period("M")})
                .groupby(["ticker", "month"]).size().rename("events")
                .reindex(pd.MultiIndex.from_product([
                    panel["tickers"], pd.period_range(min(a["dates"][0] for a in panel["arrays"]),
                                                      max(a["dates"][-1] for a in panel["arrays"]), freq="M")],
                    names=["ticker", "month"]), fill_value=0).reset_index())
    coverage["month"] = coverage["month"].astype(str)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, model, history in (("price", price, price_history), ("llm", full, full_history),
                                 ("embedding", embedding, embedding_history)):
        torch.save(model.state_dict(), out / f"{name}.pt")
        history.to_csv(out / f"{name}_history.csv", index=False)
    for name, filename in zip(names, ("predictions_llm.csv", "predictions_embedding.csv",
                                      "predictions_price.csv", "predictions_zero.csv")):
        predictions[name].to_csv(out / filename, index=False)
    signal.to_csv(out / "signal_metrics.csv")
    portfolio.to_csv(out / "portfolio_metrics.csv")
    equity.to_csv(out / "equity_curves.csv")
    split_coverage.to_csv(out / "split_coverage.csv", index=False)
    coverage.to_csv(out / "event_coverage_by_ticker_month.csv", index=False)
    pd.DataFrame([audit]).to_csv(out / "extraction_audit.csv", index=False)
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for name, color in zip((*names, "Comprar e manter"), (GOLD, CYAN, PURPLE, WHITE, GREEN)):
        ax.plot(equity.index, equity[name], color=color, linewidth=2,
                linestyle="--" if name == "Comprar e manter" else "-", label=name)
    ax.axhline(1, color=WHITE, alpha=0.2, linestyle="--")
    ax.set_title("Manchetes verificadas · carteira líquida no teste", color="white", fontsize=16)
    ax.set_ylabel("Capital (base 1; custos incluídos)", color="white", fontsize=14)
    ax.tick_params(colors="white", labelsize=12)
    ax.grid(alpha=0.12)
    for spine in ax.spines.values():
        spine.set_color(WHITE)
    ax.legend(fontsize=14, labelcolor="white", facecolor=PANEL, edgecolor=WHITE)
    fig.tight_layout()
    fig.savefig(out / "llm_vs_controls.png", dpi=300, facecolor=BG, edgecolor="none")
    plt.close(fig)
    (out / "results.md").write_text(
        "# Fatos de manchetes por LLM — avaliação diagnóstica fora da amostra\n\n"
        f"Modelo: `{MODEL_ID}` revisão `{MODEL_REVISION}`, quantização 4-bit; extração somente de títulos "
        "verificados, nunca corpos de artigos. Indícios descrevem o papel do emissor nomeado, não "
        "o retorno esperado. Campos na ordem: " + ", ".join(LLM_FEATURE_NAMES) + ". "
        "Abstenções/zeros são desconhecidos; saída inválida não é fato e fica zero. "
        "MiniLM congelado. Comparação incremental justa: mesma arquitetura, títulos, novidade, "
        "idade, máscaras, amostras e janelas; só os campos LLM são removidos da ablação. "
        "Novidade usa até dez títulos anteriores de datas de disponibilidade anteriores; "
        "até oito títulos dos últimos cinco pregões, atraso de disponibilidade de um pregão. "
        "Sem título recente, ambos os modelos de atenção preveem excesso zero. "
        "Treino/validação/teste são temporais com purga do alvo nas fronteiras; treino e parada "
        "antecipada dos modelos de atenção usam linhas com título na validação. \n\n"
        "## Auditoria da extração\n\n```\n" + pd.DataFrame([audit]).to_string(index=False) + "\n```\n\n"
        f"Cache bruto/auditável da extração: `{Path(facts_path).name}`; cache MiniLM: "
        f"`{Path(embedding_path).name}`. Inválidas acima de 5% impedem treino; porcentagens "
        "de papel, classe, direção e certeza desconhecidos contam apenas linhas válidas.\n\n"
        "## Limitações\n\n" + panel["source_note"] + "\n\n"
        "O pré-treino de Qwen2.5 (2024/2025) pode conter conhecimento dos resultados 2020–2023, "
        "mesmo com entrada somente de títulos; instrução de extrair fatos não garante ausência "
        "de vazamento de memória paramétrica. O teste 2022-09 a 2023-11 já foi inspecionado na "
        "experiência anterior: comparação diagnóstica, não confirmação independente. "
        "Universo ex post (inclusive ETFs) sujeito a viés de seleção/sobrevivência; ausência de "
        "manchete não comprova ausência de notícia. Janeiro de 2024 não tem cobertura auditada; "
        "não há avaliação de 2024+. Horizontes quinzenais consecutivos se sobrepõem; "
        "diferenças de MSE/IC ou carteiras não provam causalidade, desempenho futuro ou alpha.\n\n"
        f"## Protocolo\n\nAlvo: retorno de cinco pregões menos retorno médio do universo. "
        f"Custo {cost_bps} bps por valor negociado (compra e venda). "
        "Carteiras positivas totalmente investidas rebalanceadas a cada cinco pregões após o fechamento, "
        "sem previsão completa na data mantêm os pesos. Comprar e manter compra pesos iguais "
        "na data inicial e não rebalanceia; todas têm mesmas datas, custos de entrada e retornos. "
        "Alpha anualizado é intercepto dos retornos diários vs comprar e manter, beta é "
        "covariância/variância. MSE cobre todos os exemplos; `mse_com_evento` cobre apenas "
        "exemplos com título recente. Zero excesso é previsão nula explícita.\n\n"
        "## Cobertura por split\n\n```\n" + split_coverage.to_string(index=False) + "\n```\n\n"
        "## Métricas de previsão\n\n```\n" + signal.to_string() + "\n```\n\n"
        "## Carteiras líquidas\n\n```\n" + portfolio.to_string() + "\n```\n\n"
        "Checkpoints, históricos, previsões por método, cobertura por ticker/mês e curva no mesmo diretório.\n",
        encoding="utf-8")
    return {"signal": signal, "portfolio": portfolio, "audit": audit,
            "split_coverage": split_coverage, "coverage": coverage, "equity": equity,
            "histories": {"price": price_history, "llm": full_history, "embedding": embedding_history},
            "llm_cache": facts_path, "embedding_cache": embedding_path, "output": out}
