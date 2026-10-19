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
