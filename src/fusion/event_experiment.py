"""Colab-only out-of-sample experiment: frozen price baseline plus event residual."""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.backtest.visualization import BG, PANEL, GOLD, CYAN, WHITE, PURPLE
from src.fusion.event_model import (EventDataset, EventResidual, PriceOnly,
                                    load_article_embeddings, load_event_panel)


def _train(model, train_ds, val_ds, device, residual=False, epochs=35, patience=6):
    train = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True)
    val = DataLoader(val_ds, batch_size=256, pin_memory=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-4, weight_decay=1e-4)
    best, weights, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train:
            mom, vol, ret, text, mask, target = (x.to(device, non_blocking=True) for x in batch)
            if residual and not mask.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = model(mom, vol, ret, text, mask)[:, 0] if residual else model(mom, vol, ret)[0][:, 0]
            active = mask.any(dim=1) if residual else torch.ones_like(target, dtype=torch.bool)
            loss = torch.nn.functional.mse_loss(prediction[active], target[active])
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            losses.append((loss.item(), int(active.sum())))
        model.eval()
        vtotal, count = 0.0, 0
        with torch.inference_mode():
            for batch in val:
                mom, vol, ret, text, mask, target = (x.to(device, non_blocking=True) for x in batch)
                active = mask.any(dim=1) if residual else torch.ones_like(target, dtype=torch.bool)
                if not active.any():
                    continue
                prediction = model(mom, vol, ret, text, mask)[:, 0] if residual else model(mom, vol, ret)[0][:, 0]
                vtotal += torch.nn.functional.mse_loss(prediction[active], target[active], reduction="sum").item()
                count += int(active.sum())
        if not count or not losses:
            raise ValueError("Sem exemplos com notícia no treino/validação para avaliar o residual")
        val_mse = vtotal / count
        history.append({"epoch": epoch + 1, "train_mse": sum(v * n for v, n in losses) / sum(n for _, n in losses),
                        "val_mse": val_mse})
        if val_mse < best - 1e-8:
            best, weights, stale = val_mse, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(weights)
    return pd.DataFrame(history)


def _scores(model, dataset, device, residual=False):
    model.eval()
    results, masks = [], []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=256, pin_memory=True):
            mom, vol, ret, text, mask, _ = (x.to(device, non_blocking=True) for x in batch)
            score = model(mom, vol, ret, text, mask) if residual else model(mom, vol, ret)[0]
            results.append(score[:, 0].cpu().numpy())
            masks.append(mask.any(dim=1).cpu().numpy())
    return np.concatenate(results), np.concatenate(masks)


def _prediction_frame(panel, samples, scores, has_news):
    rows = []
    for (ti, j), score, present in zip(samples, scores, has_news):
        ti, j = int(ti), int(j)
        a = panel["arrays"][ti]
        rows.append((panel["tickers"][ti], a["dates"][j], a["target"][j],
                     a["fwd_return"][j], a["market_fwd_return"][j], float(score), bool(present)))
    return pd.DataFrame(rows, columns=["ticker", "date", "excess_return_5d", "asset_return_5d",
                                       "market_return_5d", "score", "has_event"])


def _signal_metrics(frame):
    y, p = frame["excess_return_5d"].to_numpy(), frame["score"].to_numpy()
    ic = frame.groupby("date")[["score", "excess_return_5d"]].apply(
        lambda day: day["score"].corr(day["excess_return_5d"], method="spearman")
        if len(day) > 2 and day["score"].nunique() > 1 else np.nan,
    ).mean()
    news = frame["has_event"].to_numpy()
    return {"mse_5d": float(np.mean((p - y) ** 2)), "mae_5d": float(np.mean(np.abs(p - y))),
            "ic_spearman": float(ic), "n_amostras": len(frame), "com_evento": int(news.sum()),
            "mse_com_evento": float(np.mean((p[news] - y[news]) ** 2)) if news.any() else np.nan}


def _daily_returns(panel, start):
    # All dates and all assets must be observed; never fill a missing price with zero.
    tickers = panel["tickers"]
    cols = [pd.Series(a["known_ret"], index=pd.DatetimeIndex(a["dates"]), name=ticker)
            for ticker, a in zip(tickers, panel["arrays"])]
    full = pd.concat(cols, axis=1).loc[pd.Timestamp(start):]
    if full.isna().any().any() or not np.isfinite(full.to_numpy()).all():
        raise ValueError("Preços diários incompletos no teste; comparação BH requer calendário comum")
    return full


def _tilt(scores):
    # Softmax of bounded cross-sectional z-scores: fully invested, strictly long-only.
    z = (scores - scores.mean()) / max(scores.std(), 1e-6)
    weights = np.exp(np.clip(0.35 * z, -2, 2))
    return weights / weights.sum()


def _portfolio(panel, predictions, cost_bps):
    tickers = panel["tickers"]
    score_matrix = predictions.pivot(index="date", columns="ticker", values="score").reindex(columns=tickers)
    common = score_matrix.dropna().sort_index()
    if common.empty:
        raise ValueError("Teste sem datas com previsões de todos os ativos")
    daily = _daily_returns(panel, common.index[0])
    # The five-session horizon requires decisions five sessions before the final close.
    dates = daily.index
    eligible = common.index.intersection(dates[:-5])
    if not len(eligible):
        raise ValueError("Teste curto demais para rebalanceamento quinzenal")
    first = eligible[0]
    last_rebalance = eligible[-1]
    interval = dates[(dates >= first) & (dates <= last_rebalance)]
    scheduled = interval[::5]
    rebalance = scheduled.intersection(common.index)
    skipped = set(scheduled.difference(rebalance))
    final_date = dates[dates.get_loc(rebalance[-1]) + 5]
    daily = daily.loc[first:final_date]
    if np.any(~np.isfinite(daily.to_numpy())):
        raise ValueError("Retornos diários não finitos")
    target = {date: _tilt(common.loc[date].to_numpy(dtype=float)) for date in rebalance}
    equal = np.full(len(tickers), 1 / len(tickers))
    weights, bh_weights = None, None
    capital, bh_capital = 1.0, 1.0
    series = []
    for date in daily.index:
        if weights is not None:
            # A decision at today's close cannot earn today's close-to-close return.
            returns = daily.loc[date].to_numpy(dtype=float)
            gross = 1 + float(weights @ returns)
            bh_gross = 1 + float(bh_weights @ returns)
            if gross <= 0 or bh_gross <= 0:
                raise ValueError("Retorno de carteira inválido")
            capital *= gross
            bh_capital *= bh_gross
            weights = weights * (1 + returns) / gross
            bh_weights = bh_weights * (1 + returns) / bh_gross
        turnover = 0.0
        bh_turnover = 0.0
        if date in target:
            proposed = target[date]
            turnover = np.abs(proposed - (np.zeros_like(proposed) if weights is None else weights)).sum()
            capital *= 1 - cost_bps * turnover / 10_000
            weights = proposed
            if bh_weights is None:
                bh_turnover = 1.0
                bh_capital *= 1 - cost_bps / 10_000
                bh_weights = equal.copy()
        series.append((date, capital, bh_capital, turnover, bh_turnover, date in skipped))
    return pd.DataFrame(series, columns=["date", "strategy", "bh", "turnover", "bh_turnover",
                                         "skipped_rebalance"]).set_index("date")


def _portfolio_metrics(curve, cost_bps):
    strat = curve["strategy"] / curve["strategy"].shift(fill_value=1) - 1
    bh = curve["bh"] / curve["bh"].shift(fill_value=1) - 1
    variance = float(bh.var(ddof=1))
    beta = float(strat.cov(bh) / variance) if variance > 0 else np.nan
    alpha = float((strat.mean() - beta * bh.mean()) * 252) if np.isfinite(beta) else np.nan
    final = curve.iloc[-1]
    return {"return_pct": (float(final["strategy"]) - 1) * 100,
            "bh_return_pct": (float(final["bh"]) - 1) * 100,
            "excess_pct": (float(final["strategy"]) - float(final["bh"])) * 100,
            "alpha_annual_pct": alpha * 100, "beta": beta,
            "sharpe": float(np.sqrt(252) * strat.mean() / strat.std(ddof=1)) if strat.std(ddof=1) > 0 else np.nan,
            "turnover_total": float(curve["turnover"].sum()), "cost_bps_per_traded_notional": cost_bps,
            "rebalance_count": int((curve["turnover"] > 0).sum()),
            "skipped_rebalances": int(curve["skipped_rebalance"].sum())}


def _plot(curves, path):
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    colors = {"BH": WHITE, "Preço": GOLD, "Evento": CYAN, "Evento embaralhado": PURPLE}
    for name, curve in curves.items():
        col = "bh" if name == "BH" else "strategy"
        ax.plot(curve.index, curve[col], lw=2 if name == "Evento" else 1.5,
                color=colors[name], label=name, alpha=.95 if name == "Evento" else .8)
    ax.set_title("Sinal de eventos · 5 pregões · teste fora da amostra", color="white", fontsize=16)
    ax.set_ylabel("Capital (base 1; custos incluídos)", color="white", fontsize=14)
    ax.tick_params(colors="white", labelsize=12)
    ax.grid(alpha=.12)
    ax.legend(facecolor=PANEL, labelcolor="white", edgecolor=WHITE, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=BG, edgecolor="none")
    plt.close(fig)


def run_experiment(panel_path, news_path, out_dir, cache_dir=None, cost_bps=10):
    """Train on Colab GPU; persist out-of-sample reports, curves and model weights."""
    if not torch.cuda.is_available():
        raise RuntimeError("Ative uma GPU no Colab; treino/forward local são proibidos")
    if cost_bps < 0 or cost_bps >= 10_000:
        raise ValueError("Custo de transação em bps inválido")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    panel = load_event_panel(panel_path)
    events, cache_path = load_article_embeddings(news_path, cache_dir or out / "article_cache")
    train_ds, val_ds = (EventDataset(panel, events, split) for split in ("train", "val"))
    test_ds = EventDataset(panel, events, "test")
    shuffled = EventDataset(panel, events, "test", shuffled=True)
    device = torch.device("cuda")
    baseline = PriceOnly(panel["arrays"][0]["mom"].shape[1]).to(device)
    hist_price = _train(baseline, train_ds, val_ds, device)
    torch.save(baseline.state_dict(), out / "price_only.pt")
    event_model = EventResidual(baseline, event_dim=test_ds.dim).to(device)
    hist_event = _train(event_model, train_ds, val_ds, device, residual=True, epochs=25)
    torch.save(event_model.state_dict(), out / "event_residual.pt")
    price, news_mask = _scores(baseline, test_ds, device)
    event, _ = _scores(event_model, test_ds, device, residual=True)
    permuted, shuffled_mask = _scores(event_model, shuffled, device, residual=True)
    predictions = {
        "Preço": _prediction_frame(panel, test_ds.samples, price, news_mask),
        "Evento": _prediction_frame(panel, test_ds.samples, event, news_mask),
        "Evento embaralhado": _prediction_frame(panel, test_ds.samples, permuted, shuffled_mask),
    }
    signal = pd.DataFrame({name: _signal_metrics(frame) for name, frame in predictions.items()}).T
    curves = {name: _portfolio(panel, frame, cost_bps) for name, frame in predictions.items()}
    curves = {"BH": curves["Preço"], **curves}
    portfolio = pd.DataFrame({name: _portfolio_metrics(curve, cost_bps)
                              for name, curve in curves.items() if name != "BH"}).T
    months = pd.period_range("2020-01", "2023-11", freq="M")
    coverage = (pd.DataFrame({"ticker": events["ticker"], "month": pd.to_datetime(events["decision_date"]).to_period("M")})
                .groupby(["ticker", "month"]).size().rename("events")
                .reindex(pd.MultiIndex.from_product([panel["tickers"], months], names=["ticker", "month"]), fill_value=0)
                .reset_index())
    coverage["month"] = coverage["month"].astype(str)
    signal.to_csv(out / "signal_metrics.csv")
    portfolio.to_csv(out / "portfolio_metrics.csv")
    coverage.to_csv(out / "event_coverage_by_ticker_month.csv", index=False)
    pd.DataFrame({name: curve["strategy"] if name != "BH" else curve["bh"]
                  for name, curve in curves.items()}).to_csv(out / "equity_curves.csv")
    hist_price.to_csv(out / "price_history.csv", index=False)
    hist_event.to_csv(out / "event_history.csv", index=False)
    _plot(curves, out / "event_vs_bh.png")
    (out / "results.md").write_text(
        "# Sinal de notícias por evento — experimento 2020–2023\n\n"
        "Alvo: retorno simples futuro de 5 pregões do ativo menos a média igualmente ponderada "
        "dos retornos de 5 pregões dos ativos na data. Previsões do teste nunca participaram do ajuste; "
        "encoder MiniLM congelado, baseline numérica congelada durante treino residual. "
        "Texto somente de manchetes verificadas; artigos brutos têm corpos trocados com manchetes/URLs. "
        "Data de disponibilidade da manchete atrasada um pregão; janela recente de cinco pregões.\n\n"
        "## Limitações da fonte\n\n" + panel["source_note"] + "\n"
        "Ausência de evento não demonstra ausência de notícia; períodos de feed sem cobertura não são "
        "avaliáveis. Janeiro de 2024 ausente; 2024+ não é avaliado. Universo de 49 ativos escolhido "
        "ex post, sujeito a viés de seleção/sobrevivência (inclui ETFs). "
        "Horizontes consecutivos se sobrepõem; IC e ablação não estabelecem causalidade.\n\n"
        f"## Execução\n\nManifesto/cache: `{cache_path.name}`. "
        f"Custo: {cost_bps} bps por valor nocional negociado (compras + vendas); "
        "BH compra pesos iguais no início e conserva as quantidades, sem rebalancear; "
        "sinal tenta redistribuir pesos positivos a cada 5 pregões, marcados a mercado diariamente; "
        "sem previsão completa no pregão agendado, mantém os pesos até o próximo rebalanceamento. "
        "Ambas as curvas partem da mesma data e incluem o custo de entrada no capital inicial; "
        "retornos totais são calculados desde capital 1 antes dos custos. Alpha anualizado é "
        "intercepto de retornos diários simples contra BH, beta é covariância/variância. "
        "A ablação desloca eventos a datas passadas e troca tickers, sem permitir notícias futuras.\n\n"
        "## Métricas de sinal\n\n```\n" + signal.to_string() + "\n```\n\n"
        "## Carteira fora da amostra\n\n```\n" + portfolio.to_string() + "\n```\n\n"
        "Cobertura por ticker/mês: `event_coverage_by_ticker_month.csv`. "
        "Curvas na mesma janela de datas: `equity_curves.csv`, `event_vs_bh.png`.\n",
        encoding="utf-8",
    )
    return {"signal": signal, "portfolio": portfolio, "coverage": coverage,
            "curves": curves, "price_history": hist_price, "event_history": hist_event,
            "cache": cache_path, "output": out}
