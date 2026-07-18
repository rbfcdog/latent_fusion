# TODO

## Active Priorities

- [ ] Implement directional attention, alignment, and total loss as described in the paper
- [ ] Implement TabPFN embedding module for comparison
- [ ] Compare with models like XGBoost and NLP-only approaches
- [ ] HMM (Markov switching model) for detecting regime changes (breakpoints)
- [ ] More complex quant concepts

## Research Tracks

- [ ] **Reinforcement Learning** — RL agent on embeddings for policy optimization
- [ ] **Transformer** — attention-based architectures for text × TS fusion
- [ ] **Crypto** — test strategies on DRW crypto dataset, vol surface, HFT 15m–1h

## Backtest Engine (src/backtest/) ✅

- [x] **BacktestEngine** — signals → equity curve, fees, slippage, full metrics
- [x] **Strategies** — SMA cross, mean reversion, HMM regime, VWAP reversion, Institutional V3
- [x] **Monte Carlo** — N perturbed paths, noise + delay, percentile fan bands
- [x] **Parameter grid search** — exhaustive param sweep
- [x] **Walk-forward optimization** — rolling train/validation/test, purge + embargo
- [x] **Stress tests** — COVID, 2008, 2022, flash crash, volmageddon, dot-com
- [x] **Dark-mode visualization** — equity curve, MC fan, stress bars, heatmap, WF timeline

### Metrics Implemented

- [x] **Sharpe ratio** (annualized)
- [x] **Sortino ratio**
- [x] **CAPM alpha & beta** (vs benchmark)
- [x] **CAGR** (compound annual growth rate)
- [x] **Max drawdown** (peak-to-trough)
- [x] **Calmar ratio**
- [x] **Win rate, profit factor, trade count**

## Data Pipeline

- [x] LSE B3 price data — 26 tickers OHLCV (`data/lse_market_data/b3/`)
- [x] Google News RSS headlines — 26 tickers, 12K+ articles (`data/news_b3_lse/`)
- [x] News content scraping — RSS summary → markdown (`scripts/scrape_news_content.py`)
- [ ] Test strategy on B3 data (VWAP reversion or Inst V3 on Brazilian tickers)
- [ ] Test strategy on DRW crypto dataset
- [ ] Merge with `/quant/` repo gradually

## Research Track (Details)

- [ ] Separate alpha/beta
- [ ] Reinforcement learning on embeddings
- [ ] Cross-reference outliers
- [ ] Volatility surface
- [ ] Hawkes process
- [ ] HMM

## From Chat (2026-05-22)

- [ ] Investor profile (long-term, quarterly rebalance constraints)
- [ ] Better embedding models + numerical models
- [ ] HMM × embedding alignment
- [ ] Extract value from embeddings (not just direction)
- [ ] Actually **train** the model (not just inference)
- [ ] RL + optimization policy based on profile
- [ ] Brazil dataset (news + TS) for B3

## UX / Product

- [ ] Train from signals with profile-based regularization + granularity (include taxes)
- [ ] Better UX for NuBank integration

## Visualization & Quant

- [ ] Focus on what makes sense: Hawkes process, realized vol 3D graph, textual info
- [ ] Re-investigate quant: best algorithms + viz (**focus heavily on viz**)
- [ ] Evaluate manual trading viability

## Live Trading

- [ ] Execute **one** trade with the strategy:
  - [ ] Run on PC
  - [ ] Consider TP/SL and leverage
  - [ ] Separable alpha
  - [ ] Real-time data (crypto options, volume-based)
  - [ ] Vol surface (implied vol)
  - [ ] Benchmark vs buy-and-hold
  - [ ] Focus HFT on 15m–1h candles
- [ ] Build model for B3 (1m timeframe) for NuBank



---

## Reunião Allan — Julho 2026

Sprint de execução: iterar por longo tempo, um item por vez.

### R1 — Rebalanceamento semanal ✅

- [x] Adicionar `rebalance_freq` ao `BacktestConfig` ("daily", "weekly", "monthly")
- [x] Engine só executa trades nos dias de rebalanceamento (sinal entre datas é mantido)
- [x] Testar impacto de weekly vs daily em Sharpe, turnover e custo

### R2 — Perfis de investimento (arrojado / moderado / seguro) ✅

- [x] Criar `InvestorProfile` com parâmetros: `max_leverage`, `max_position`, `vol_target`, `drawdown_limit`
- [x] Adequar modelo ao perfil: clipping de sinal, sizing por vol, stop por drawdown
- [x] Backtest com 3 perfis no mesmo dataset e comparar métricas

### R3 — Custo operacional e compliance no backtest ✅

- [x] Adicionar `cost_model` ao `BacktestConfig`: fee_bps + slippage_bps + spread_bps + tax_rate (IR/IOF Brasil)
- [x] Calcular custo total por trade (fee + slippage + spread + imposto proporcional)
- [x] Reportar `total_cost_pct` e `net_return_pct` nos métricas

### R4 — K-Fold e Monte Carlo embutidos no BacktestEngine ✅

- [x] Método `engine.k_fold_cross_validation(df, strategy, k=5)` com split temporal estratificado
- [x] Método `engine.monte_carlo(df, strategy, n_sims=200)` integrando `monte_carlo.py`
- [x] Retornar métricas agregadas (mean ± std) e bandas percentis

### R5 — Split estatístico de tickers ✅

- [x] Análise estatística antes do split: retorno médio, vol, correlação, liquidez, drawdown
- [x] Estratificar split para balancear tickers de desempenho desigual entre treino/teste
- [x] Evitar viés: tickers com performance extrema concentrados só no teste

### R6 — Controle de tickers e análise de pior caso ✅

- [x] Métricas por ticker no resultado do backtest (não só agregado)
- [x] Análise de pior caso: pior ticker, pior periodo, pior drawdown individual
- [x] Identificar tickers que degradam o portfólio e opção de exclusão automática

### R7 — Modelos lineares e MLP alongside embeddings textuais ✅

- [x] Expandir `TrainedEmbeddingStrategy` para ElasticNet, MLP, Ridge, Lasso, XGBoost
- [x] Rodar comparação sistemática: embeddings textuais vs features técnicas vs fusão
- [x] Tabela de resultados: correlação, excess return, alpha por modelo

### R8 — Pipeline para cripto (10+ ativos) ✅

- [x] Puxar notícias e preços para ≥10 ativos cripto (BTC, ETH, SOL, etc.)
- [x] Adaptar pipeline para variações extremas de preço (robust scaling, vol clipping)
- [x] Backtest em dados cripto com mesmo framework do B3/NASDAQ

### R9 — Carteira híbrida (B3 + Cripto + NASDAQ) ✅

- [x] Combinar dados de B3, cripto e NASDAQ em pipeline unificado
- [x] Puxar notícias das 3 fontes com alinhamento temporal
- [x] Backtest de carteira híbrida com alocação cross-asset

### R10 — Experimentos mais realistas ✅

- [x] Simular latência de execução, parcial fills, gap de overnight
- [x] Incluir restrições reais: horário de mercado, liquidez mínima, tamanho de ordem
- [x] Avaliar viabilidade como estratégia de trade aplicável

### R11 — Docs: nuvem barata + inferência em tempo real ✅

- [x] Documentar deploy em VPS low-cost (Hetzner/DigitalOcean) com systemd
- [x] Guia de inferência em tempo real: polling, webhook, streaming
- [x] Estimar custo mensal de infraestrutura por ativo monitorado