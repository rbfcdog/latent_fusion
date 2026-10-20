# News sources for the fusion experiment

## Evaluation period

Use a contiguous period with a reliable news feed, not only trading days with articles. Selecting only news days biases evaluation toward event-heavy periods. Within a covered period, retain both news and no-news days and report their results separately.

None of the 49 local raw news files contains January 2024 articles. Treat this as a feed gap, not evidence that no news occurred. Exclude the gap from the primary text evaluation. If it cannot be repaired, report a contiguous pre-gap test as a limited result, not evidence of performance in 2024–25. Preserve chronological train/validation/test splits and purge forward-label windows at split boundaries.

## Candidate sources

| Source | Use | Limitation |
| --- | --- | --- |
| [GDELT](https://www.gdeltproject.org/data.html) | Free historical article discovery and archive searches | Company matching needs verification; indexed records do not guarantee usable full article text. |
| [Massive stock news](https://massive.com/docs/rest/stocks/news) | Ticker-linked historical titles, descriptions, and publication timestamps | Its documentation lists two years of history on Stocks Basic and all history on Stocks Starter; verify access and article-text licensing before committing. |
| [SEC EDGAR](https://www.sec.gov/edgar/sec-api-documentation) | Historical filings and company-event text | Filings are not a substitute for daily news; evaluate them as a separate source. |

First sample January 2024 for a few tickers. Check publication timestamps, ticker/entity relevance, duplicate URLs, and trading-day coverage. If a source works, backfill the entire study period consistently rather than patching January alone: a provider change can otherwise look like a change in the text signal. Keep unknown feed coverage distinct from genuinely quiet news days; lag text when same-day availability is uncertain.

## Audited event dataset for the Colab signal test

The 49 local ticker files contain 124,596 tagged records dated January 2020–November 2023. Some `Article` bodies do not match their own `Article_title` and URL, so **do not embed the bodies**. `scripts/export_event_signal.py` retains 27,654 distinct headlines only after verifying that title and URL agree and that the headline identifies the tagged issuer or fund. This is a high-precision subset, **not** a complete news archive; no event in the export does not establish that no news existed.

The exporter writes gitignored `data/fusion_events.jsonl.gz` (one timestamped, source-attributed headline per event) and `data/fusion_event_panel.npz` (cached MOMENT plus a five-trading-day excess-return target). It maps publication to the first trading day on or after its date and delays availability by one additional trading day. It limits the experiment to **2020-01-01 through 2023-11-30** rather than treating the 2024 feed failure as quiet news. The available GDELT DOC request for January 2024 returned HTTP 429; there is no configured historical paid-news API key. Backfill remains unresolved.

The Colab notebook `notebooks/05_fusion/event_signal_colab.ipynb` uses these two exported files, trains only on Colab GPU, and compares an event-gated residual against the same-date price-only portfolio and equal-weight buy-and-hold after costs. The ticker universe was selected retrospectively by news coverage and includes ETFs: selection bias and event relevance remain research limitations.

The Colab panel loader preserves `datetime64[ns]` trading dates. Portfolio calendar checks must compare prediction and price dates at the same resolution; truncating one side to day precision falsely reports missing sessions.

## Text-first attention experiment (Colab)

Run `notebooks/05_fusion/text_attention_colab.ipynb` on a Colab GPU using the same audited `fusion_events.jsonl.gz` and `fusion_event_panel.npz` already in `MyDrive/latent_fusion_data/`. It reuses the cached MiniLM title vectors and writes **new** reports/checkpoints to `MyDrive/latent_fusion_data/text_attention_results/`; no local fusion run is required.

Each title contributes its frozen embedding, 17 bounded, interpretable cues (event class, explicit earnings/guidance/rating language, observed percentage, update/rumor/uncertainty), age and novelty against up to ten earlier-available titles for the same ticker. These cues describe the headline, **not necessarily the tagged company's role**; percentages are not analyst-consensus surprises. Same-day headlines do not establish novelty for one another. Numeric history queries self-attended title events, but cannot reach the prediction head without a recent event. A mask/age-only attention control tests whether the title content adds information beyond event timing; price-only, zero-excess prediction and buy-and-hold are separate baselines.

In a data-only check of all 27,654 audited titles, 22,821 fell outside the five explicit event classes and 3,211 matched earnings/guidance. This sparse taxonomy is an extra cue, not a complete language representation; the embedding remains available to attention. Train/validation/test follow the existing temporal, purged panel split. The 2022–23 test was examined in the previous experiment, so new results on it remain exploratory; an independently gathered post-gap news period is necessary before a confirmatory performance claim.

## LLM headline extraction and attention ablation (Colab)

Run `notebooks/05_fusion/llm_attention_colab.ipynb` on a Colab T4/L4 GPU with the **same** two audited Drive artifacts. It downloads the pinned, [Apache-2.0 Qwen2.5-7B-Instruct checkpoint](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) in 4-bit mode, so the first run needs substantial time and disk space. No external LLM API key or local inference is required. The per-title JSONL cache in `MyDrive/latent_fusion_data/llm_headline_cache/` resumes after Colab disconnects; rerun the extraction cell until the whole manifest is complete, then run the training cell. Outputs go to a separate `llm_attention_results/` folder.

The LLM sees only each verified title plus the tagged ticker and its known aliases. It returns issuer role, event class, explicit direction and certainty, and a literal percentage with an exact title quotation. Unknown/ambiguous roles abstain; malformed or unsupported responses are saved with their raw output and `status=invalid`, not passed off as facts. The notebook displays an extraction audit and refuses to train when more than 5% of titles have invalid responses. A valid `unknown` is distinct from an invalid response.

The experiment holds the numeric-query attention architecture, title dates, novelty, and event masks fixed while comparing frozen MiniLM **with and without** these LLM facts; it also reports price-only, zero-excess and buy-and-hold controls on the same test period. A better classification of headline wording does **not** establish a better forecast. The Qwen checkpoint postdates the 2020–23 headlines and may encode later outcomes despite title-only prompting; the already-inspected 2022–23 test is exploratory. A new, consistently covered period is needed for confirmation.

## Fast fusion presentation demo (Colab)

`notebooks/05_fusion/fast_fusion_demo_colab.ipynb` replaces the slow 7B LLM extraction with a small, genuine multimodal attention comparison. It uses the **existing Drive panel for dates and observed returns only**, not its stored MOMENT vectors. On each Colab GPU run it generates fresh MOMENT-1-small embeddings from the previous 64 known daily returns for every required numeric input and fresh MiniLM embeddings for **all 8,455 audited titles** of eight fixed assets (ADBE, BABA, COST, CVX, GILD, NKE, PYPL, QCOM). The verified titles already have the conservative one-session delay; days without titles remain in the evaluation. No separate price CSV upload or old article-embedding cache is required.

On Python 3.13, `momentfm==0.1.4` cannot resolve its pinned NumPy 1.25.2 and Transformers 4.33.3 dependencies. The notebook installs MOMENT without those dependency pins, keeps a supported Transformers 4.x release, and checks the MOMENT import before extraction. This version combination is not upstream-supported; verify the actual GPU run in Colab before presenting its results.

The notebook trains at most eight epochs on the chronological, purged panel and compares numeric-only against numeric-query cross-attention over headlines. It saves newly inferred embedding arrays, model identities, generation timestamp, signal/coverage metrics, long-only backtests after 10 bps, an eight-asset buy-and-hold curve, and a zero-excess **forecast** reference in `MyDrive/latent_fusion_data/fast_fusion_demo_results/`. The target is five-session forward return relative to the **same eight-asset basket** plotted as buy-and-hold, not relative to the earlier 49-asset universe.

This is a smaller **demonstration**, not a replacement for the 49-asset MOMENT-large study or proof of profitable text. The asset subset was chosen ex post for news coverage, 2022–23 test outcomes were already examined, and no 2024+ source gap is filled. Generation time depends on the Colab GPU and model downloads; it avoids the much slower 7B token-by-token decoding rather than promising a fixed runtime.
