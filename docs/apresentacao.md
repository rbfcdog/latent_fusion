# Apresentação — Latent Fusion

> Roteiro de fala para os notebooks `portifolio.ipynb` e `final_report.ipynb`.
> Todas as estratégias rodam sobre o **mesmo portfólio cross-sectional de 49 ações US** sob condições idênticas de mercado.

---

## 1. Motivação e Tese Central

> "A pergunta central do projeto é: embeddings de texto de notícias financeiras carregam informação que não está nos preços? E se carregam, dá pra transformar isso em alpha real?"

A tese em uma linha:

**Deriva de embedding → proxy de chegada de informação → intensidade Hawkes → regime HMM → tilt cross-sectional → portfólio**

---

## 2. Dados (`portifolio.ipynb`)

> "Trabalhamos com os 50 tickers de maior cobertura jornalística no nosso corpus de notícias."

| Métrica | Valor |
|---|---|
| Tickers no universo | 50 |
| Embeddings de texto (all-MiniLM-L6-v2, 384-dim) | 122.250 |
| Janela histórica | 2009-04-14 → 2025-05-06 |
| Painel de retornos `ret_df` | (3.849 dias × 49 tickers) |
| Intensidade Hawkes `intensity_df` | (3.849 × 49) |
| Split treino / teste | 2.694 / 1.155 dias (70/30 temporal) |

> "O split é estritamente temporal — nunca embaralhamos. O modelo nunca vê o futuro."

---

## 3. Pipeline de Features (`portifolio.ipynb`)

> "A partir dos embeddings, calculamos deriva diária por ticker:"

$$\Delta_{t,i} = \|E_{t,i} - E_{t-1,i}\|_2$$

> "Quando a deriva supera o percentil 95 do treino, registramos um evento. Esses eventos alimentam uma intensidade estilo Hawkes:"

$$\lambda_{t,i} = e^{-\beta}\lambda_{t-1,i} + \alpha \cdot \mathbf{1}\{\Delta_{t,i} \geq q_{0.95}\}$$

> "O **gate hard-70** seleciona só os tickers nos 30% mais intensos — os que tiveram mais chegada de informação nova naquele dia. O regime de volatilidade determina se aplicamos momentum ou reversão à média."

A melhor configuração no treino: **`S1_hard70_mom3_rev`** — TrainSharpe 0.835, TestSharpe 1.041.

---

## 4. Modelo HMM (`portifolio.ipynb`)

> "Treinamos um HMM gaussiano de 3 estados sobre retorno médio e intensidade média do portfólio."

| Estado | Retorno médio | Intensidade média | Regime |
|---|---|---|---|
| 0 | +0.000743 | 0.277 | Bull calmo — momentum |
| 1 | −0.000469 | 0.641 | Crise / alta incerteza — reversão |
| 2 | +0.000461 | 0.070 | Drift silencioso — baixa atividade |

- Log-likelihood treino: **11.745**
- Regimes altamente persistentes: prob. auto-transição ~98% para estados 0 e 2
- Distribuição treino: 1.663 / 104 / 927 dias → o estado de crise é raro e curto

> "O modelo identifica automaticamente quando o mercado está em crise versus bull market, e alterna a lógica do sinal de acordo."

---

## 5. Validação Cross-Sectional por Ticker (`portifolio.ipynb`)

> "Para testar robustez além do tempo, dividimos os tickers: 24 para treino, 25 para teste. O modelo nunca viu os tickers de teste."

- Tickers treino: CMG, V, XLE, USO, CVX, MRK, KO... (24 total)
- Tickers teste: AAL, ABBV, ABT, AMT, BBY... (25 total)

> "O sinal gerado nos tickers de teste mantém performance positiva. Isso prova **generalização cross-sectional** — não só temporal."

---

## 6. Fusão Texto + MOMENT (`portifolio.ipynb`)

> "A segunda modalidade usa MOMENT-1-large — embeddings numéricos de 1.024 dimensões sobre janelas de séries de preço e indicadores técnicos."

Quando fundimos as duas modalidades via HMM conjunto (3 componentes PCA de texto + 3 de MOMENT):

| Modelo | Retorno total | BH | Excess | Sharpe |
|---|---|---|---|---|
| **HMM Texto + MOMENT** | **+113.45%** | +84.78% | **+28.67%** | **0.940** |
| Buy & Hold | +84.78% | — | — | 0.619 |

> "Sharpe de 0.940 contra 0.619 do BH — melhora de 52%. Texto domina: contribuição de 92.6% do alpha versus 7.4% dos indicadores técnicos."

---

## 7. Estratégias no Portfólio Real (`final_report.ipynb`)

> "No `final_report`, todas as estratégias — determinísticas e baseadas em embedding — rodam sobre o mesmo portfólio cross-sectional de 49 ações, no mesmo período de teste, contra o mesmo benchmark."

Essa é a comparação justa: **mesmas condições de mercado para todos**.

### Estratégias determinísticas (portfólio)

| Estratégia | Retorno | Sharpe | Alpha |
|---|---|---|---|
| SMA 20/100 | +80.6% | 2.10 | −18.1% |
| SMA 10/50 | +154.0% | 3.05 | −7.3% |
| Mean Reversion | −60.4% | −2.98 | −6.7% |
| Regime Router | −10.9% | −0.22 | — |
| S1 Hard70 | — | — | — |
| Intensity Gated | −4.8% | −0.23 | +0.60% |

> "SMA tem Sharpe alto porque captura bem o trend. Mas o alpha negativo mostra que não bate o portfólio de mercado em retorno ajustado ao risco."

### Estratégias com embeddings (portfólio, cross-sectional)

| Estratégia | Retorno | BH | Excess | Sharpe | Alpha |
|---|---|---|---|---|---|
| **Text Embedding S1 Hard70** | **+122.4%** | +94.3% | **+28.1%** | 0.447 | **+0.91%** |
| MOMENT (numérico) | +26.0% | +20.97% | +5.1% | 0.705 | +0.78% |

> "As duas estratégias baseadas em embedding são as únicas com **excess return positivo E alpha CAPM positivo**. O texto carrega informação que não está nos preços."

---

## 8. Modelos Supervisionados (`final_report.ipynb`)

> "Além dos modelos determinísticos, treinamos modelos supervisionados usando os embeddings como features para prever retornos futuros."

| Modelo | Retorno teste | BH | Excess | Sharpe |
|---|---|---|---|---|
| Ridge | +116.2% | +94.3% | +22.0% | 1.000 |
| **Lasso** | **+118.7%** | +94.3% | **+24.4%** | **1.019** |
| ElasticNet | +118.1% | +94.3% | +23.9% | 1.015 |

> "Lasso tem o maior excess. A esparsidade L1 ajuda — os pesos de embedding relevantes são poucos. 189 das 384 dimensões capturam 80% do poder preditivo."

---

## 9. Grid Search (`final_report.ipynb`)

> "Varredura exaustiva de hiperparâmetros no treino para 4 famílias de estratégias: SMA Cross, Mean Reversion, Regime Router e S1 Hard70."

- Os heatmaps mostram Sharpe por combinação de parâmetros sobre o portfólio real
- Contraste automático: texto branco em células escuras, escuro em claras
- A melhor estratégia por Sharpe no grid search seleciona automaticamente os parâmetros para Monte Carlo, stress test e walk-forward

---

## 10. Validação Temporal — 5 Folds (`final_report.ipynb`)

> "5 folds com janelas expansivas sobre o portfólio. A cada fold o modelo é re-otimizado com mais dados — simula produção real sem data leakage."

- Fold size: ~500 dias de treino crescendo até ~2.700
- Métricas registradas por fold: Sharpe, excess return, alpha, max drawdown
- Estratégias com alpha médio positivo ao longo dos folds têm maior robustez fora de amostra

---

## 11. Monte Carlo — Robustez do Sinal (`final_report.ipynb`)

> "200 simulações perturbando o sinal com ruído e delay de 0 a 3 dias."

| Percentil | Equity final |
|---|---|
| p5 | 13.610 |
| Mediana | 13.784 |
| Média | 13.754 |
| p95 | 13.831 |
| Base | 13.832 |

> "Variância pequena entre os 200 paths — std ~83 sobre capital de 10.000. O sinal não depende de timing perfeito."

---

## 12. Análise de Risco (`final_report.ipynb`)

### Tail risk — Estratégia vs Buy & Hold

| Métrica | Estratégia | BH |
|---|---|---|
| Pior dia | −3.03% | −3.03% |
| VaR 95% | **−1.41%** | −1.65% |
| VaR 99% | −2.31% | −2.48% |
| ES 95% | −1.94% | −2.17% |
| Kurtosis | 2.94 | −0.19 |
| Skew | 0.23 | 0.04 |

> "VaR 95% melhor que o BH: a estratégia tem caudas ligeiramente menores. Kurtosis positiva — distribuição leptocúrtica, esperado em dados financeiros diários."

### Regime breakdown

| Regime | Sharpe estratégia | Sharpe BH |
|---|---|---|
| Alta volatilidade | **0.90** | menor |
| Baixa volatilidade | menor | menor |

> "A estratégia performa melhor em alta volatilidade — faz sentido: o gate Hawkes é ativado por eventos intensos, que ocorrem mais em crises."

### Rolling Sharpe (63 dias)

- Média: 1.21 — Std: 2.25 — Máximo: 5.55 — Mínimo: −3.11

---

## 13. Contribuição por Ticker (`final_report.ipynb`)

> "Top 5 contribuidores do alpha de texto no período de teste."

- **Top:** BX, USO, COP, HPQ, GE — setores de energia e financeiro geraram mais sinal
- **Bottom:** BIIB, PYPL — setores de biotech e fintech drenaram alpha

> "Isso é informativo: o modelo de notícias captura bem eventos corporativos discretos em energia e financeiro, onde anúncios têm impacto imediato e previsível."

---

## 14. Conclusões

**1. Embeddings de texto geram alpha cross-sectional real.**
S1 Hard70 entrega +28.1% de excess return e alpha CAPM de +0.91% sobre o portfólio de mercado. Confirmado por Ridge/Lasso (+24.4% excess, Sharpe >1.0).

**2. Fusão de modalidades melhora Sharpe.**
Texto + MOMENT via HMM conjunto: Sharpe 0.940 vs 0.619 do BH — melhora de 52%.

**3. Generalização dupla: temporal e cross-sectional.**
O sinal funciona em períodos fora de amostra E em tickers nunca vistos no treino.

**4. Sinal robusto a execução imperfeita.**
Monte Carlo 200 paths com delay 0–3 dias: variância <1% do capital. Pode operar com latência.

---

## Cola rápida — respostas para perguntas

| Pergunta | Resposta |
|---|---|
| Quantos tickers? | 49–50, top por cobertura de notícias |
| Quantos embeddings? | 122.250 de texto (384-dim) + MOMENT (1.024-dim) por ticker |
| Janela histórica | 2009–2025 texto; 2020–2025 MOMENT |
| Split treino/teste | 70/30 temporal — sem shuffle |
| Todas as estratégias na mesma base? | Sim — mesmo portfólio, mesmo benchmark, mesmo período |
| Melhor excess return | +28.1% (Text Embedding S1 Hard70) |
| Melhor Sharpe embedding | 0.940 (HMM Texto + MOMENT) |
| Melhor modelo supervisionado | Lasso: Sharpe 1.019, excess +24.4% |
| Quantas dimensões importam? | 189/384 capturam 80% do poder preditivo |
| Texto vs técnicos | 92.6% texto, 7.4% indicadores técnicos |
| Monte Carlo | 200 paths, delay 0–3 dias, std final <1% do capital |
| VaR 95% | −1.41% estratégia vs −1.65% BH |
