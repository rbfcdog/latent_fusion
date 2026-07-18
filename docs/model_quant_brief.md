# Quant Review Packet: Latent Fusion Portfolio Models

This document summarizes the notebooks from a quant-modeling perspective so another LLM, researcher, or reviewer can critique the design. The goal is not to sell the model; it is to make the data flow, assumptions, equations, and validation risks explicit.

Source artifacts reviewed:

- `EDA.ipynb`: market-data EDA, realized volatility, sentiment checks, MOMENT/time-series embedding exploration, GARCH/HMM sketches, attention-analysis guide.
- `embeddings.ipynb`: sparse news alignment, sentence embeddings, kernel aggregation, embedding HMM regimes, sentiment-only baseline, FiLM/gating fusion, volatility prediction.
- `portifolio.ipynb`: daily text embeddings, embedding-drift intensity, deterministic and HMM-routed portfolio strategies, model comparison, text+numerical HMM strategy.
- `portifolio.md`: conceptual portfolio-theory framing: Markowitz, Black-Litterman, risk parity, embeddings as views/confidence.

## Review Prompt For Another LLM

Please review this pipeline as a quant strategy. Focus on:

1. Whether the timing assumptions avoid lookahead bias.
2. Whether embedding drift is a defensible proxy for information arrival.
3. Whether the Hawkes-style intensity process is statistically justified or should be estimated.
4. Whether HMM regimes are used in a tradable way.
5. Whether the validation design supports generalization across time and tickers.
6. Whether the portfolio construction preserves alpha or washes it out through clipping and normalization.
7. What baselines, transaction-cost tests, and ablations are missing.

## One-Line Model Summary

The implemented strategy is an event-driven, long-only cross-sectional allocator: text embeddings are converted into a persistent news-intensity state, that state gates price momentum/reversal rules, and the resulting alpha surface tilts an equal-weight buy-and-hold portfolio.

High-level pipeline:

$$
\text{prices/news}
\rightarrow \text{EDA and universe filter}
\rightarrow \text{daily text embeddings}
\rightarrow \text{embedding drift}
\rightarrow \text{event intensity}
\rightarrow \text{regime/gates}
\rightarrow \text{cross-sectional tilt}
\rightarrow \text{portfolio returns}
$$

## Notation And Timing

Let:

- $i \in \{1,\dots,N\}$ index tickers.
- $t \in \{1,\dots,T\}$ index daily dates.
- $P_{t,i}$ be close price.
- $r_{t,i}=P_{t,i}/P_{t-1,i}-1$ be close-to-close return.
- $E_{t,i}\in\mathbb{R}^D$ be the text embedding for ticker $i$ on date $t$.
- $m_{t,i}\in\{0,1\}$ be the availability mask.
- `split` is usually the first 70% of dates for training and the last 30% for testing.

The portfolio notebook builds:

$$
r^{fwd}_{t,i}=r_{t+1,i}
$$

and stores it in `ret_df` after merging on `(ticker, date)`. Strategy PnL at date $t$ is computed using `ret_df[t]`, so the intended decision convention is:

> Use information available on date $t$ to hold from $t$ close to $t+1$ close.

Critical timing assumption: all news embedded into $E_{t,i}$ must be known before the trade that earns $r^{fwd}_{t,i}$. If news timestamps are not filtered by market close, the text features should be lagged by one trading day.

## EDA Layer

The EDA notebook establishes whether the data are usable before modeling.

Main tasks:

1. Load `data/time_series/*.csv` and `data/text/*.jsonl`.
2. Build a price panel for up to the first 100 tickers.
3. Compute daily simple returns and 20-day annualized volatility:

$$
\text{vol}_{20,t,i}=\sqrt{252}\cdot \operatorname{std}(r_{t-19:t,i})
$$

4. Rank assets by volatility, return distribution, and data coverage.
5. Compute realized-volatility breakout events:

$$
RV_{t,i}=\sqrt{\sum_{s=t-w+1}^{t} \left(\log(P_{s,i}/P_{s-1,i})\right)^2}
$$

6. Run a sentiment check with FinBERT-style positive/neutral/negative probabilities.
7. Compare sentiment with future returns.
8. Explore MOMENT time-series embeddings and clustering.
9. Test GARCH/HMM regime sketches on high-volatility names.

Important EDA conclusion from the notebook narrative:

> Sentiment-only features show weak near-zero correlation with future returns in the sampled data, so the project moves away from raw sentiment classification and toward richer embedding-state variables.

Quant interpretation: EDA is not the alpha model. It defines the tradable universe, checks sparsity, and motivates why raw sentiment is too weak to use alone.

## Universe Construction

The portfolio notebook uses a news-coverage filter:

1. Count unique news days per ticker from `data/text/*.jsonl`.
2. Select the top `N`, usually `top_N = 50`.
3. Cache the ranking in `data/top_50_news_stocks.csv`.

Reason: text embeddings are sparse. A top-news universe reduces missingness and improves cross-sectional comparability.

Reviewer concern: selecting the universe using full-sample news coverage may introduce survivorship/selection bias. A stricter design would select names using only pre-train information or rebalance the universe through time.

## Text Embedding Layer

### Daily ticker documents

For each ticker-day, the notebook aggregates up to:

- `max_articles_per_day = 5`
- `max_chars_per_day = 8000`

The merged document is embedded with:

```text
SentenceTransformer("all-MiniLM-L6-v2")
```

The resulting panel is:

$$
E_{t,i}\in\mathbb{R}^D
$$

with metadata `(ticker, date)` cached in:

- `cache/text/top50_daily_embeddings.npy`
- `cache/text/top50_daily_metadata.csv`

Quant interpretation: this is not a direct return predictor. It is a latent representation of the daily news state.

### Kernel aggregation for sparse news

The embeddings notebook also handles irregular event times by projecting event embeddings onto a daily grid:

$$
z_t^{(\lambda)}
=
\frac{
\sum_{e:\tau_e\le t} \exp(-\lambda(t-\tau_e))z_e
}{
\sum_{e:\tau_e\le t} \exp(-\lambda(t-\tau_e)) + \epsilon
}
$$

where:

- $z_e$ is an event/article embedding.
- $\tau_e$ is the event timestamp.
- $\lambda$ controls decay.
- tested values include $\lambda\in\{0.01,0.05,0.10\}$.

Quant interpretation: kernel aggregation creates a continuous daily text context from sparse news arrivals. Smaller $\lambda$ means longer memory; larger $\lambda$ reacts more to recent news.

## Technical / Time-Series Embedding Layer

The notebooks use `MOMENTPipeline` from `AutonLab/MOMENT-1-large` with `task_name="embedding"` to embed market time-series windows.

In `portifolio.ipynb`, the implemented multi-channel technical window uses:

- `Close`
- `rsi_14`
- `atr_14`
- `bb_bandwidth`
- `adx`

For each ticker and date, a trailing window is prepared:

- `window_days = 30`
- `min_window_len = 5`

The window is padded if needed and passed to MOMENT:

$$
X_{t,i}^{tech}\in\mathbb{R}^{C\times L}
\rightarrow
M_{t,i}\in\mathbb{R}^{d_M}
$$

In `embeddings.ipynb`, a broader technical-indicator section explores more channels such as SMA/EMA/HMA, RSI, stochastic indicators, ATR, ADX, MACD, Bollinger/Keltner/Donchian channels, rolling volatility, and related features.

Quant interpretation: MOMENT is used to convert recent market structure into a dense numerical state. In the current portfolio strategy, the main traded alpha still comes from text-derived intensity and lagged returns; MOMENT is more developed in the fusion and volatility-prediction experiments.

## Embedding Drift And Event Detection

The portfolio notebook aligns daily embeddings to forward returns and constructs a ticker-date panel.

Missing embedding values are filled by:

1. Reindexing to the full `(date, ticker)` grid.
2. Filling missing embeddings within each date by that date's cross-sectional mean.
3. Filling remaining missing values with zero.

Then embeddings are standardized using train-split statistics only:

$$
\tilde E_{t,i,d}=\frac{E_{t,i,d}-\mu_d^{train}}{\sigma_d^{train}}
$$

Embedding drift is:

$$
\Delta_{t,i}
=
\left\|\tilde E_{t,i,:}-\tilde E_{t-1,i,:}\right\|_2
$$

A binary information event is triggered using the 95th percentile of train drift:

$$
e_{t,i}
=
\mathbf{1}\left\{\Delta_{t,i}\ge q_{0.95}(\Delta_{train})\right\}
$$

Quant interpretation: large embedding movement is treated as an abnormal information-arrival event.

Reviewer concerns:

- L2 drift in embedding space may not correspond to economic novelty.
- Cross-sectional mean imputation can dampen or distort drift.
- The 95th percentile threshold is heuristic, not estimated from a point-process model.
- Embeddings should be lagged if same-day text availability is uncertain.

## Hawkes-Style Intensity

The event stream is smoothed into an intensity state:

$$
\lambda_{t,i}
=
\exp(-\beta_h)\lambda_{t-1,i}
+
\alpha_h e_{t,i}
$$

with:

$$
\alpha_h=0.8,\qquad \beta_h=0.2
$$

Interpretation:

- An event creates an immediate intensity jump.
- Intensity decays exponentially without new events.
- Multiple event days cluster into a persistent high-news-activity state.

This is called Hawkes-style because it resembles self-exciting event memory, but the parameters are fixed. It is not a full Hawkes model estimated by likelihood.

Reviewer concern: if the paper or report calls this a Hawkes process, it should either estimate $\alpha_h,\beta_h$ with a likelihood or explicitly call it a deterministic Hawkes-inspired filter.

## Regime Models

The notebooks contain multiple regime variants.

### 1. Deterministic volatility regime

The simplest regime is built from lagged market-average returns:

$$
\bar r^{known}_t=\frac{1}{N}\sum_i \text{known\_ret}_{t,i}
$$

where:

$$
\text{known\_ret}_{t,i}=\text{ret\_df}_{t-1,i}
$$

Then:

$$
\sigma^{mkt}_{20,t}
=
\operatorname{std}_{20}(\bar r^{known}_t)
$$

and:

$$
R_t=
\mathbf{1}\left\{
\sigma^{mkt}_{20,t} >
\operatorname{median}(\sigma^{mkt}_{20,train})
\right\}
$$

This regime is used to route momentum versus reversal signals.

### 2. HMM on average return and intensity

The portfolio notebook fits a 3-state Gaussian HMM:

$$
x_t=
\begin{bmatrix}
\bar r_t\\
\bar\lambda_t
\end{bmatrix}
,\qquad
z_t\in\{1,2,3\}
$$

with:

$$
x_t\mid z_t=k\sim\mathcal{N}(\mu_k,\Sigma_k)
$$

where:

$$
\bar r_t=\frac{1}{N}\sum_i r_{t,i},
\qquad
\bar\lambda_t=\frac{1}{N}\sum_i \lambda_{t,i}
$$

State quality is ranked by:

$$
q_k=
\frac{\mu_{k,return}}{\mu_{k,intensity}+10^{-6}}
$$

The best state is used as a favorable regime.

Important caveat: in some notebook cells, this HMM uses `ret_df`, which contains forward returns. If HMM states inferred from `ret_df[t]` are used to trade `ret_df[t]`, this is lookahead. For a tradable model, the HMM feature should use only lagged returns, e.g. `known_ret`, plus intensity known at decision time.

### 3. Ticker-split HMM validation

The notebook splits tickers with seed 42:

$$
\mathcal{I}_{train}\cap\mathcal{I}_{test}=\emptyset
$$

It fits the HMM on train-ticker aggregate features, then evaluates portfolios separately on train and test tickers.

Purpose: test cross-sectional generalization.

Reviewer concern: the HMM state prediction in that cell uses full-universe average return features in places. If those returns are forward returns, the test is not clean. The ticker split is a good idea, but the regime inference must be made with lagged/known features.

### 4. Text plus numerical HMM from 2020 onward

Another portfolio cell builds an HMM from:

Text features:

- daily mean text embeddings
- interpolate/bfill/ffill to the return dates
- train-standardize
- reduce to 3 PCs

Numerical market features:

- market mean known return
- market absolute return
- breadth
- market intensity mean
- market intensity std
- intensity 5-day momentum
- market return 5-day momentum
- market return 10-day momentum
- market 20-day volatility

These are train-standardized and reduced to 3 PCs. The HMM input is:

$$
X_t =
\left[
PC^{text}_{1:3,t},
PC^{num}_{1:3,t}
\right]
\in\mathbb{R}^6
$$

Then:

$$
X_t\mid z_t=k\sim\mathcal{N}(\mu_k,\Sigma_k)
$$

State quality is based on buy-and-hold return in the train window, and the chosen state routes reversal versus momentum.

This variant is closer to a tradable regime model because the numerical features are lagged/known features rather than target returns.

## Signal Construction

### Common building blocks

Intensity rank:

$$
p_{t,i}=\operatorname{rankpct}(\lambda_{t,i})
$$

Soft gate:

$$
g^{soft}_{t,i}
=
\operatorname{clip}(2(p_{t,i}-0.5),0,1)
$$

Hard gate:

$$
g^{70}_{t,i}
=
\mathbf{1}\{p_{t,i}\ge 0.70\}
$$

Intensity acceleration:

$$
a^{accel}_{t,i}
=
\max\left(
0,
\operatorname{zscore}_{cross-section}
(\lambda_{t,i}-\lambda_{t-1,i})
\right)
$$

Reversal:

$$
rev_{t,i}
=
-\operatorname{sign}(\text{known\_ret}_{t,i})
$$

Momentum:

$$
mom^{(h)}_{t,i}
=
\operatorname{sign}
\left(
\operatorname{MA}_h(\text{known\_ret}_{t,i})
\right)
$$

with horizons such as $h=3,5,10,20$.

### Deterministic strategy signals

The implemented deterministic signal family includes:

$$
S1_{t,i}
=
g^{70}_{t,i}
\left[
R_t rev_{t,i}
+
(1-R_t)mom^{(3)}_{t,i}
\right]
$$

$$
S2_{t,i}
=
g^{soft}_{t,i}
\left[
R_t rev_{t,i}
+
(1-R_t)mom^{(5)}_{t,i}
\right]
$$

$$
S3_{t,i}
=
a^{accel}_{t,i}
\left[
R_t rev_{t,i}
+
(1-R_t)mom^{(10)}_{t,i}
\right]
$$

$$
S4_{t,i}
=
g^{65}_{t,i}
(1-R_t)mom^{(5)}_{t,i}
$$

Interpretation:

- In high-volatility or favorable HMM regimes, the strategy often uses short-term reversal.
- Outside those regimes, it uses momentum.
- Embedding intensity gates determine which assets receive active tilts.

## Portfolio Construction

The baseline is equal-weight over available assets:

$$
w^{BH}_{t,i}
=
\frac{m_{t,i}}{\sum_j m_{t,j}}
$$

Raw signal $a_{t,i}$ is converted into a market-neutral cross-sectional tilt:

$$
\tilde a_{t,i}
=
\frac{a_{t,i}-\bar a_t}
{\sum_j |a_{t,j}-\bar a_t|}
$$

Then the long-only portfolio is:

$$
\hat w_{t,i}
=
\max\left(0,\; w^{BH}_{t,i}+k\tilde a_{t,i}\right)
$$

$$
w_{t,i}
=
\frac{\hat w_{t,i}}{\sum_j \hat w_{t,j}}
$$

The grid searches:

$$
k\in\{0.05,0.10,0.15,0.20,0.30,0.40,0.50\}
$$

Exposure overlay:

$$
E_t=
\begin{cases}
up, & \text{if } \operatorname{MA}_{20}(\bar r^{known}_t)>0\\
down, & \text{otherwise}
\end{cases}
$$

with:

$$
up\in\{1.00,1.10,1.20,1.30\},\qquad
down\in\{0.20,0.40,0.60,0.80,1.00\}
$$

Daily strategy return:

$$
r^p_t
=
E_t\sum_i w_{t,i}r^{fwd}_{t,i}
$$

The strategy remains long-only in asset weights. The overlay scales gross exposure in the return calculation.

Reviewer concerns:

- Long-only clipping can remove negative alpha views and dilute the intended signal.
- Normalizing after clipping makes the active exposure state-dependent.
- The code does not yet explicitly model turnover, slippage, taxes, borrow costs, or leverage financing.
- If the exposure overlay is later normalized away in plotted weights, plotted weights may not reflect return exposure.

## Train Objective And Model Selection

For each signal and hyperparameter tuple, the train score is:

$$
\text{TrainScore}
=
\text{Sharpe}_{train}
+
0.5\cdot \text{TotalReturn}_{train}
-
0.25\cdot \text{Vol}_{train}
$$

The selected model is the highest train-score model. Test outperformance versus buy-and-hold is reported as a diagnostic:

$$
\Delta_{test}
=
\text{TotalReturn}_{test}
-
\text{TotalReturn}^{BH}_{test}
$$

Reviewer concern: because multiple signal families and grids are tried, test results should be treated as research diagnostics unless there is a separate untouched validation period or nested walk-forward design.

## Model Comparison Family

`portifolio.ipynb` compares these raw alpha models under the same portfolio construction:

1. Price mean reversion:

$$
a_{t,i}=-\operatorname{sign}(\text{known\_ret}_{t,i})
$$

2. Price momentum 5d:

$$
a_{t,i}=\operatorname{sign}(\operatorname{MA}_5(\text{known\_ret}_{t,i}))
$$

3. Price momentum 20d:

$$
a_{t,i}=\operatorname{sign}(\operatorname{MA}_{20}(\text{known\_ret}_{t,i}))
$$

4. Low-volatility momentum:

$$
a_{t,i}=g^{lowvol}_{t,i}mom^{(10)}_{t,i}
$$

5. Embedding intensity reversion:

$$
a_{t,i}=g^{70}_{t,i}rev_{t,i}
$$

6. Embedding intensity acceleration plus momentum:

$$
a_{t,i}=a^{accel}_{t,i}mom^{(10)}_{t,i}
$$

7. Embedding regime-routed:

$$
a_{t,i}
=
g^{soft}_{t,i}
\left[
R_t rev_{t,i}
+
(1-R_t)mom^{(5)}_{t,i}
\right]
$$

8. Prior optimized embedding model, if `best_ret` exists.

This comparison is useful because it separates price-only baselines from embedding-conditioned variants.

## Metrics

The notebook reports:

- Total return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Calmar ratio
- Tracking error versus buy-and-hold
- Information ratio
- Annualized CAPM alpha
- CAPM beta
- Correlation with benchmark
- Hit rate

CAPM alpha/beta are estimated as:

$$
r^p_t-r_f
=
\alpha
+
\beta(r^{BH}_t-r_f)
+
\epsilon_t
$$

In code:

$$
\alpha_{ann}=252\cdot
\left(
\mathbb{E}[r^p-r_f]
-
\beta\mathbb{E}[r^{BH}-r_f]
\right)
$$

Rolling alpha and beta use a 63-day window.

## Embedding HMM Versus Sentiment-Only Baseline

The embeddings notebook compares:

- HMM using kernel-aggregated text embeddings plus returns and volatility.
- HMM using FinBERT-style sentiment probabilities only.

It also separates beta and alpha components with Ridge regressions:

Beta features:

$$
X^\beta_t=
[
r_{t-1},
\sigma_{t-1},
\text{regime one-hot}_{t-1}
]
$$

Alpha residual features:

$$
X^\alpha_t=
[
\|z_t-z_{t-1}\|_2,
\text{semantic volatility}_t,
\text{sentiment entropy}_t
]
$$

The check asks whether text-derived features explain residual return variation beyond market beta/regime features.

Quant interpretation: this is an alpha-separation diagnostic, not yet a fully robust trading model.

## Multimodal Fusion Experiments

`embeddings.ipynb` sketches two fusion mechanisms between numerical/MOMENT embeddings and text embeddings.

### FiLM

$$
h_t=\text{MLP}(x^{num}_t)
$$

$$
\gamma_t,\beta_t=g(z^{text}_t)
$$

$$
x^{fused}_t=\gamma_t\odot h_t+\beta_t
$$

Interpretation: text modulates the numerical representation by scaling and shifting features.

### Gating

$$
g_t=\sigma(Wz^{text}_t)
$$

$$
x^{fused}_t=g_t\odot x^{num}_t
$$

Interpretation: text controls how much of the numerical state to trust.

These fused embeddings are tested with Ridge regression to predict 20-day realized volatility. This is a research prototype for a supervised multimodal model, separate from the current deterministic portfolio strategy.

## Portfolio-Theory Interpretation

The conceptual `portifolio.md` frames embeddings in three possible portfolio roles.

### Markowitz

Classical objective:

$$
\max_w
\quad
w^\top \mu
-
\frac{\lambda}{2}w^\top\Sigma w
$$

Embedding role:

$$
\mu_i=f(E_{t,i},X^{tech}_{t,i})
$$

or embeddings define semantic clusters used as concentration penalties.

### Black-Litterman

Prior:

$$
\mu\sim\mathcal{N}(\pi,\tau\Sigma)
$$

Views:

$$
P\mu=q+\epsilon,\qquad \epsilon\sim\mathcal{N}(0,\Omega)
$$

Embedding role:

$$
q_i=f(E_{t,i}),\qquad
\Omega_{ii}=1/\text{confidence}_i
$$

Main insight: embeddings may be better used as noisy views plus confidence estimates than as direct return forecasts.

### Risk parity

Embeddings can adjust risk:

$$
\sigma_{i,adj}
=
\sigma_i(1+\text{semantic\_risk}_i)
$$

This is not fully implemented in the current backtest but is a plausible next step.

## What Is Actually Implemented Versus Experimental

Implemented in portfolio backtests:

- top-news universe selection
- daily sentence embeddings
- embedding drift
- binary event threshold
- Hawkes-style intensity recursion
- deterministic volatility regime
- HMM variants
- cross-sectional momentum/reversal signal gates
- long-only tilted portfolio
- temporal and ticker-split diagnostics
- price-only versus embedding model comparison

Experimental / exploratory:

- FinBERT sentiment-only baseline
- kernel-aggregated embedding HMM on one aligned ticker context
- MOMENT technical embeddings
- FiLM/gating multimodal fusion
- Ridge volatility prediction
- Black-Litterman and risk-parity integration
- cross-attention interpretability guide

## Main Quant Hypothesis

The strategy assumes:

1. News embeddings encode economically relevant information.
2. Large day-to-day embedding changes proxy for information shocks.
3. Information shocks cluster and persist, so an intensity memory is useful.
4. Cross-sectional intensity rank is more robust than raw intensity level.
5. The effect of a news shock depends on market regime:
   - some regimes favor mean reversion after shocks;
   - other regimes favor continuation/momentum.
6. A constrained long-only tilt around buy-and-hold is more stable than a fully directional long/short model.

## Main Risks To Review

### Timing and lookahead

- Confirm that all news in $E_{t,i}$ is available before earning $r^{fwd}_{t,i}$.
- Any HMM that uses `ret_df[t]` as a state feature for trading `ret_df[t]` is not tradable.
- Use `known_ret` or lagged realized returns for regime inference.

### Universe and sample selection

- Top-news tickers are selected using full-sample coverage.
- This can bias toward assets that were active or survived through the sample.
- A production design should form the universe using only information available at each rebalance date.

### Missing data

- Cross-sectional mean imputation may hide no-news days.
- Zero filling may create artificial low-drift states.
- Need ablation: no imputation, carry-forward embedding, explicit no-news indicator.

### Statistical estimation

- Hawkes parameters are fixed, not estimated.
- HMM states may be unstable across seeds and samples.
- PCA dimension choices are heuristic.
- Signal thresholds such as 65%, 70%, and 95% are heuristic.

### Backtest realism

- No explicit transaction costs.
- No turnover constraint.
- No tax model.
- No liquidity/capacity model.
- No exchange calendar handling described in the brief.
- No slippage or borrow/leverage financing.

### Multiple testing

- Several models, signals, hyperparameters, and diagnostic plots are tried.
- A separate holdout or walk-forward validation is needed before claiming alpha.

## Recommended Ablations

1. Price-only baseline with identical portfolio construction.
2. Randomized embeddings preserving date/ticker sparsity.
3. Lagged text embeddings by one trading day.
4. No-Hawkes version using raw event indicators only.
5. Estimated Hawkes parameters versus fixed parameters.
6. No-regime version.
7. Deterministic volatility regime versus HMM regime.
8. Sentiment-only versus embedding-only versus text+price.
9. No cross-sectional imputation.
10. Turnover-penalized weights.
11. Transaction-cost stress at several bps assumptions.
12. Walk-forward hyperparameter selection.

## Suggested Next Quant Architecture

The most defensible next version is:

1. Define a realistic rebalance calendar and data timestamp convention.
2. Lag all text features conservatively.
3. Estimate an expected-return view and confidence from embeddings:

$$
q_{t,i}=f(E_{t,i},\lambda_{t,i},R_t)
$$

$$
c_{t,i}=g(E_{t,i},\text{news coverage}_{t,i},\lambda_{t,i})
$$

4. Feed those into Black-Litterman:

$$
\Omega_{t,ii}=1/(c_{t,i}+\epsilon)
$$

5. Use a covariance model with shrinkage.
6. Optimize with turnover, long-only, sector/cluster, and liquidity constraints.
7. Evaluate with walk-forward splits and transaction costs.

This would align better with the insight in `portifolio.md`: embeddings should likely generate uncertain views, not directly determine raw portfolio weights.

## Bottom Line

The notebooks implement a hybrid quant pipeline, not a pure NLP classifier. The core tradable idea is:

$$
\text{text embedding change}
\rightarrow
\text{information intensity}
\rightarrow
\text{regime-conditioned momentum/reversal}
\rightarrow
\text{long-only portfolio tilt}
$$

The most important review question is whether the timing is clean. If text features and HMM states are properly lagged, the model becomes an interpretable event-driven allocator. If any same-day forward return enters regime inference or feature construction, the reported alpha can be materially overstated.
