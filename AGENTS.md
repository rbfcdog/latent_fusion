# Latent Fusion — Multimodal Financial Prediction

Research project (UNICAMP IC) fusing text embeddings with time-series data for
regime-conditioned portfolio allocation.

## Stack

- Python 3.11+, PyTorch, Jupyter notebooks
- `sentence-transformers` (all-MiniLM-L6-v2) for daily news embeddings
- `MOMENT-1-large` for time-series embeddings
- `hmmlearn` for Hidden Markov Model regime detection
- Package manager: `uv` (pyproject.toml + uv.lock)
- Environment: `.venv/`

## Repo Structure

```
latent_fusion/
├── src/                                  # Production-grade library
│   ├── backtest/                         # Full backtest engine
│   │   ├── engine.py                     # BacktestEngine: signals → equity curve
│   │   ├── monte_carlo.py               # Monte Carlo: noise + delay perturbation
│   │   ├── optimizer.py                  # Grid search + walk-forward optimization
│   │   ├── stress_test.py               # COVID, 2008, 2022, flash crash scenarios
│   │   └── visualization.py             # Dark-mode plotting (equity, MC fan, heatmap)
│   ├── strategy/                         # All trading strategies
│   │   └── strategies.py                 # SMA, MeanRev, HMM, VWAP, InstV3, RegimeRouter, S1Hard70
│   └── features/                         # Feature engineering
│       ├── indicators.py                 # SMA, EMA, RSI, ATR, ADX, MACD, Bollinger, etc.
│       └── volatility.py                 # Realized vol, Hawkes, regime labels
│
├── backtest/                             # Legacy notebooks-compatible backtests
│   ├── backtest_engine.py
│   ├── regime_backtest_comparison.py
│   ├── multi_strategy_backtest.py
│   └── hmm_technical_backtest.py
│
├── models/                               # ML models & regime detection
│   ├── temporal_fusion.py
│   └── hmm_regimes.py
│
├── features/                             # Legacy features (notebook-compatible)
│   ├── technical.py
│   ├── volatility_regime_indicators.py
│   └── implied_volatility_surface.py
│
├── notebooks/                            # Jupyter notebooks (numbered by topic)
│   ├── 01_eda/                           # EDA.ipynb, EDA_test.ipynb
│   ├── 02_embeddings/                    # embeddings.ipynb
│   ├── 03_volatility/                    # Vol surface, Hawkes+HMM indicators
│   ├── 04_backtests/                     # Regime vs buy-hold, LSE, NASDAQ
│   ├── 05_fusion/                        # Attention, cross-modal experiments
│   └── 06_portfolio/                     # Portfolio construction + comparison
│
├── scripts/                              # Standalone utilities (not imported)
│   ├── scraper.py
│   ├── b3_scraper.py
│   └── get_news_temporal_alignment.py
│
├── docs/                                 # Documentation
│   ├── plan.md                           # Research plan
│   ├── model_quant_brief.md              # Full architecture review
│   ├── portifolio.md                     # Portfolio theory notes
│   └── ic___rodrigo.pdf                  # IC paper PDF
│
├── outputs/                              # Generated outputs
│   ├── html/                             # HTML reports
│   └── logs/                             # Log files
│
├── data/                                 # Raw datasets (gitignored)
├── cache/                                # Precomputed cache
├── images/                               # Generated charts, figures & animations
├── checkpoints/                          # Model checkpoints (.pt)
└── latex/                                # IC paper (main.tex)
```

## Core Thesis

> Text embedding drift → information-arrival proxy → Hawkes-style intensity
> → regime-conditioned momentum/reversal gates → long-only cross-sectional
> portfolio tilt.

Pipeline:

```
news + prices → text embeddings → embedding drift → event intensity
  → HMM regime detection → momentum/reversal signal gates
  → cross-sectional tilt → portfolio returns
```

## Critical Rules (DO NOT VIOLATE)

1. **No lookahead bias.** All features used for trading at time `t` must be
   computable from information available ≤ `t`.
   - `ret_df` = forward returns (target). Only use for evaluation, **never** as
     a feature.
   - `known_ret` = lagged return (safe feature at decision time).
   - Any HMM state inference that uses `ret_df[t]` to trade `ret_df[t]` is a
     **bug**.
2. **Temporal train/test split.** Always split by date (first 70% train, last
   30% test). Never shuffle.
3. **Long-only portfolio weights.** Clipped at ≥ 0, normalized to sum to 1
   around equal-weight baseline.
4. **News timestamps must be lagged** if same-day availability is uncertain.
   Default: lag text features by 1 day.
5. **Always compare vs market (buy & hold).** Every backtest must report:
   - Strategy return vs BH return (excess return)
   - Alpha/beta decomposition (α = strategy return − β × market return)
   - BH curve plotted alongside strategy equity curve

## Key Files

| File | Purpose |
|------|---------|
| `backtest/backtest_engine.py` | `BacktestEngine` class — walk-forward backtesting |
| `features/technical.py` | `compute_technical_indicators()`, MOMENT integration |
| `models/temporal_fusion.py` | `TemporalFusion` cross-attention model |
| `models/hmm_regimes.py` | HMM training, sentiment regime classification |
| `features/volatility_regime_indicators.py` | Hawkes intensity, realized vol, regime labels |
| `docs/model_quant_brief.md` | Full architecture review — read this first |
| `docs/plan.md` | Research plan: regime detection, text×TS fusion, ablations |
| `TODO.md` | Current priorities and backlog |

## Conventions

- Prefer notebook exploration → extract to `features/` or `models/` when a
  pattern solidifies
- **Notebook path fix.** All notebooks must start with this cell to resolve
  `src` imports from any cwd:
  ```python
  import sys, os
  from pathlib import Path
  _root = Path(os.getcwd())
  while not ((_root / 'src').exists() and (_root / 'pyproject.toml').exists()) and _root != _root.parent:
      _root = _root.parent
  sys.path.insert(0, str(_root))
  ```
- Cache expensive computations in `cache/` (embeddings, aggregations)
- Use `uv run python <script>` for script execution
- All `data/` files are gitignored (large); paths are expected to exist locally
- When adding notebooks follow the `NN_topic/` numbering convention
- **No standalone print statements in notebooks.** Only output DataFrames, Series,
  or dicts as the last expression of a cell. Never use `print()` for status
  messages like "Done", "All OK", "Imported" — let the object repr do the work.
- **Markdown cells must use correct Portuguese accentuation** (acentuação: 
  Execução, Estratégias, Métricas, Validação, Análise, Técnicos, etc.).
  Never write Portuguese words without accents in markdown.

## Visualization Rules

All charts and figures must use dark mode aesthetic. Reference style from
`backtest/multi_strategy_backtest.py` (`strategy_comparison_all.png`):

- Figure background: `#0d0d1a`, subplot backgrounds: `#1a1a2e`
- Color palette: GOLD `#D4A843`, GREEN `#00E676`, RED `#FF1744`,
  CYAN `#18FFFF`, WHITE `#AAAAAA`, PURPLE `#CE93D8`, ORANGE `#FF9800`,
  BLUE `#42A5F5`
- White ticks, subtle grid (`alpha=0.12`), dashed reference lines (`alpha=0.2`)
- Green/red `fill_between` for above/below baseline zones
- Horizontal bar charts for metric comparisons
- Use `matplotlib.use('Agg')` for non-interactive rendering
- Save with `dpi=300`, `facecolor='#0d0d1a'`, `edgecolor='none'`
- **One plot per image.** No subplots. Each saved file contains exactly one
  chart. Use separate images for separate ideas.
- **Font sizes must be legible.** Titles 16pt, axis labels 14pt,
  tick labels 12pt, legend 14pt, heatmap annotations 12pt. Nothing below 12pt.
- **Heatmap text must contrast with background.** Use white text on dark
  cells and dark text on light cells. Always include a colorbar with value
  mapping on the side.

## Code Generation Rules

When asked to generate code:

- DO NOT write explanations
- DO NOT generate Markdown explaining the code
- DO NOT describe what the code does
- DO NOT summarize the solution
- Return only the requested code
- No text before or after the code
- No comments (inline, block, or documentation)
- No debug output, prints, or logs unless explicitly required
- No emojis
- No decorative text (banners, separators, titles, section headers)
- No unnecessary imports, placeholder code, unused variables, or scaffolding
- Output only what is strictly necessary to implement the requested functionality
