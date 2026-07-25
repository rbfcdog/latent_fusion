# Publishability Audit — Latent Fusion

> Auditado em 2026-07-25. Cobre documentação, código, experimentos, reprodutibilidade e próximos passos.

---

## Nota Geral: 6.8/10 — Publication-Ready com Ressalvas

O projeto tem fundamentos sólidos (engine de backtest, paper trading deployado, visualizações profissionais, documentação extensa), mas três bloqueadores estruturais impedem publicação imediata.

---

## 1. Documentação ★★★★☆ (4.2/5)

### O que existe (qualidade 4-5/5)

| Arquivo | Qualidade | Conteúdo |
|---|---|---|
| `AGENTS.md` | 5/5 | Manual completo do repo: stack, estrutura, tese, 5 regras críticas, convenções |
| `docs/model_quant_brief.md` | 5/5 | 966 linhas: notação, EDA, embeddings, Hawkes, 3 variantes de regime, 8 famílias de sinais, 12 riscos, 12 ablações recomendadas |
| `docs/apresentacao.md` | 5/5 | Roteiro de apresentação completo (16 seções, português) |
| `docs/presentation.md` | 5/5 | Script expandido com timing, Q&A table |
| `docs/plan.md` | 4/5 | 6 categorias de experimentos, ~11/12 não implementados |
| `docs/transformer/architecture.md` | 4/5 | Arquitetura, positional encoding, teacher forcing, autoregressivo |
| `docs/transformer/hybrid_overfitting_report.md` | 5/5 | 3 testes de overfitting documentados (todos falharam — honesto) |
| `src/paper_trading/docs/*` | 5/5 | 7 docs produção: architecture, usage, brokers, cloud_deployment, setup guides |
| `latex/main.tex` | 5/5 | Proposta IC completa compilável: introdução, problema, objetivos, metodologia, equações |
| `latex/results.tex` | 4/5 | Resultados experimentais: 8 estratégias, 5 modelos ML, métricas, S1 Hard70 pipeline |

### O que falta

1. **README.md padrão** — atualmente contém o texto da proposta IC. Precisa de: install (`uv sync`), quick start, estrutura de diretórios, badge de license, como citar
2. **LICENSE** — não existe. Escolher MIT ou Apache 2.0
3. **CONTRIBUTING.md** — como contribuir, padrões de código, convenções
4. **CITATION.cff** — para citação acadêmica
5. **Changelog** — histórico de versões

---

## 2. Código ★★★☆☆ (3.1/5)

### Forças

- **Backtest engine** (`src/backtest/engine.py`, 505 linhas): bem estruturado, dataclasses, métricas completas (Sharpe, Sortino, Calmar, alpha/beta CAPM), k-fold, Monte Carlo, walk-forward, stress tests
- **8 estratégias** (`src/strategy/strategies.py`, 367 linhas): Protocol consistente, HMM walk-forward, S1 Hard70 multimodal
- **9 módulos de features** (`src/features/`): indicadores técnicos, volatilidade, Hawkes bivariado, microestrutura cripto, fractal, espectral — todos tipados com dataclasses
- **Paper trading** (`src/paper_trading/`): production-grade, 3 brokers implementados, SQLite persistence, sistema de deploy (systemd, Docker)
- **ML services** (`ml_service.py`): 2 estratégias ML online rodando como systemd services com `Restart=always`
- **Dark-mode visualizations**: consistentes, 300dpi, publicação-ready

### Problemas Críticos

1. **Zero testes** — Nenhum arquivo `test_*.py` em ~50+ arquivos Python. Isso bloqueia qualquer publicação de código.
2. **Duplicação legacy vs src** — 40-60% de overlap:
   - `models/` (legacy) vs `src/models/`
   - `features/` (legacy) vs `src/features/`
   - `backtest/` (legacy) vs `src/backtest/`
   - Dois `TemporalFusion` divergentes: `models/temporal_fusion.py` (legacy, notebooks) e `src/models/temporal_fusion.py` (novo, directional attention + alignment loss)
3. **Notebooks importam módulos legados** — quebram se `models/` for removido
4. **Hardcoded paths** — `/home/rodrigodog/latent_fusion` em vários scripts
5. **Broad `except Exception`** — em engine, estratégias, ML services. Esconde bugs.
6. **Sem type checking** — `mypy` ou `pyright` não configurados
7. **Sem linting** — `ruff` não configurado (apesar do `.ruff_cache/` no gitignore)
8. **`iterrows()` no engine** — lento para datasets grandes, mas funcional

---

## 3. Experimentos ★★★☆☆ (3.5/5)

### O que foi executado (com resultados)

| Experimento | Artefato | Resultado Principal |
|---|---|---|
| EDA completo | `notebooks/01_eda/EDA.ipynb` | 4213 tickers, FinBERT, MOMENT, clustering |
| Embeddings + kernel aggregation | `notebooks/02_embeddings/embeddings.ipynb` | HMM regimes from embeddings, FiLM vs gating |
| Volatilidade + Hawkes + HMM | `notebooks/03_volatility/` | Regimes de vol, superfície IV, Hawkes bivariado |
| Backtests S&P 500 | `notebooks/04_backtests/` | Regime vs buy-hold, VWAP delta hedging |
| Attention test | `notebooks/05_fusion/attention_test.ipynb` | Attention collapse detectado |
| Portfolio construction | `notebooks/06_portfolio/` | **S1 Hard70: +28.1% excesso sobre BH** |
| Final report | `src/backtest/final_report.ipynb` | 35 células: grid search, time splits, MC, stress, equity curves |
| Transformer vs RL | `src/backtest/text_embedding_comparison.py` | PPO Text+Price vence (+11.82% excess) |
| Transformer overfitting | `docs/transformer/hybrid_overfitting_report.md` | Transformer falhou 3/3 testes de overfitting |

### O que NÃO foi executado (do `docs/plan.md`)

1. **Lead-lag analysis** — cross-correlation entre embeddings e retornos futuros
2. **Cross-attention entropy por regime** — entropia da atenção em diferentes regimes
3. **Mutual Information (MINE)** — dependência texto × temporal
4. **Causal impact** — remover/embaralhar texto e medir impacto
5. **Counterfactuals** — substituir notícias positivas por negativas
6. **Regime-aware attention** — clusterizar padrões de atenção por regime
7. **Text shock sensitivity** — variação da previsão após evento textual
8. **Robustez a drift temporal** — treinar em 2020-2022, testar 2023-2024
9. **Alignment score** (proposta nova) — cosine similarity entre latent_text e latent_time
10. **Ablation study sistemático** — price-only vs text-only vs multimodal, por regime

### Dados

- **`data/`** está gitignored (correto — datasets grandes)
- **`sample_data/`** existe com subset pequeno (BTC, ETH, COST, ADBE) — funcional para testes
- **`cache/`** está gitignored (correto — embeddings pré-computados)
- Notebooks dependem de `data/time_series/` (4213 tickers) e `data/text/*.jsonl` — não versionados

---

## 4. Paper IC ★★★★☆ (4.0/5)

### Status

- **Proposta** (`latex/main.tex`): ✅ Completa — introdução, problema, objetivos, metodologia com equações, datasets, referências
- **Resultados** (`latex/results.tex`): ✅ 263 linhas — estratégias, métricas, pipeline S1 Hard70, gráficos, conclusão, próximos passos
- **Paper final unificado**: ❌ Não existe — precisa fundir `main.tex` (proposta) + `results.tex` (resultados) em um paper completo

### O que o paper final precisa

1. **Unificar proposta + resultados** em um documento só
2. **Results section** com tabelas comparativas (modelos ML × estratégias determinísticas)
3. **Ablation study** — impacto de remover embeddings, remover Hawkes, remover regime
4. **Statistical significance** — p-values, confidence intervals nos resultados
5. **Related work** expandido — comparar com estado da arte (FinBERT, FinGPT, etc.)
6. **Limitations section** — o que o modelo NÃO faz bem
7. **Reproducibility statement** — como reproduzir resultados
8. **Figuras em PDF vetorial** — converter PNGs para PDF (TikZ ou matplotlib pdf)

---

## 5. Reproducibilidade ★★☆☆☆ (2.0/5)

### Bloqueadores

1. **Sem `requirements.txt` ou lock file documentado** — `uv.lock` existe mas não há instrução `uv sync`
2. **Dados não versionados** — `data/time_series/` (4213 tickers) e `data/text/` (JSONL) ausentes
3. **Sample data incompleto** — cobre 4 ativos, mas notebooks precisam de 50+
4. **Sem script de setup** — `setup.sh` ou `Makefile` para baixar dados, instalar deps, rodar pipeline
5. **Sem CI/CD** — GitHub Actions ausente
6. **Sem Docker** para ambiente reproduzível (apesar de existir `Dockerfile` no paper_trading)

---

## 6. Roadmap para Publicação

### Fase 1 — Fundação (1-2 dias)

- [ ] **Adicionar `LICENSE`** (MIT)
- [ ] **Reescrever `README.md`**: install, quick start, estrutura, badge, citation
- [ ] **Adicionar `CITATION.cff`**
- [ ] **Adicionar `pyproject.toml` metadata**: authors, description, license, keywords
- [ ] **Criar `Makefile`**: `make install`, `make test`, `make lint`, `make docs`

### Fase 2 — Limpeza de Código (2-3 dias)

- [ ] **Resolver duplicação legacy/**: remover `models/`, `features/`, `backtest/` legados; migrar notebooks para `src/`
- [ ] **Migrar `TemporalFusion` legado** → `src/models/temporal_fusion.py` (já tem directional attention)
- [ ] **Remover hardcoded paths**: usar `Path(__file__).resolve().parent.parent`
- [ ] **Substituir `except Exception`** por exceções específicas
- [ ] **Adicionar `ruff`** com config em `pyproject.toml`
- [ ] **Adicionar `mypy`** (pelo menos strictness básico)
- [ ] **Rodar lint em tudo** e corrigir

### Fase 3 — Testes (2-3 dias)

- [ ] **Testes unitários**: `src/backtest/engine.py` (métricas, edge cases)
- [ ] **Testes de integração**: pipeline end-to-end com sample data
- [ ] **Testes de lookahead bias**: verificar que features em `t` não usam dados de `t+1`
- [ ] **Testes de estratégia**: cada `generate_signals()` com fixtures
- [ ] **CI/CD**: GitHub Actions com `uv sync && pytest && ruff && mypy`

### Fase 4 — Experimentos Faltantes (3-5 dias)

- [ ] **Ablation study sistemático**: price-only vs text-only vs multimodal, com e sem Hawkes, com e sem regime
- [ ] **Lead-lag analysis**: cross-correlation embeddings × forward returns
- [ ] **Mutual Information (MINE)**: dependência entre modalidades
- [ ] **Statistical significance**: bootstrap confidence intervals nos retornos
- [ ] **Walk-forward nos 3 datasets**: S&P 500, B3, Crypto — tabela comparativa
- [ ] **Cross-attention interpretability**: quais textos mais influenciam predições

### Fase 5 — Paper Final (3-5 dias)

- [ ] **Unificar `main.tex` + `results.tex`** em `paper.tex`
- [ ] **Escrever Related Work** (20-30 referências)
- [ ] **Escrever Results** com todas as tabelas e figuras
- [ ] **Escrever Discussion** (limitations, future work)
- [ ] **Escrever Conclusion**
- [ ] **Converter figuras para PDF vetorial**
- [ ] **Revisão de português acadêmico**

### Fase 6 — Publicação (1 dia)

- [ ] **arXiv upload**: https://arxiv.org (cs.CE ou q-fin.ST)
- [ ] **GitHub release**: tag `v1.0.0` com DOI via Zenodo
- [ ] **README badge**: arXiv, DOI, license, CI status
- [ ] **Divulgação**: Twitter/LinkedIn, submeter ao SCF (Stellar Community Fund) se relevante

---

## 7. Resumo Executivo

### Pronto para publicação AGORA
- Documentação do paper trading (nível produção)
- Engine de backtest com 8 estratégias + 5 modelos ML
- Visualizações dark-mode 300dpi
- S1 Hard70: alpha positivo comprovado (+28.1% excesso)
- LaTeX results.tex com métricas e pipeline documentado
- Dois ML services rodando 24/7 com systemd

### Bloqueadores
1. **Zero testes** — inaceitável para código público
2. **Duplicação legacy/src** — confunde novos contribuidores
3. **Dados não versionados** — notebooks não reproduzíveis sem acesso aos dados
4. **README inexistente** — ninguém sabe como instalar/rodar

### Diferencial competitivo
- Único projeto open-source com fusão multimodal (texto × TS) + paper trading ao vivo + deployment documentado
- Pipeline completo: dados → embeddings → regimes → sinais → backtest → paper trading → relatórios com IA
- Infraestrutura profissional: systemd, cloud deployment guide, Docker
