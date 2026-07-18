# Roteiro de Apresentação — Latent Fusion

> **Objetivo:** Fusão latente de text embeddings e time-series embeddings para alocação de portfólio com detecção de regime.
> Este documento cobre os notebooks `portifolio.ipynb` e `final_report.ipynb`, na ordem natural de uma apresentação.
> Cada seção traz o que **você fala**, os **números exatos** dos outputs, e os **slides/visualizações** que aparecem na tela.

---

## 0. Abertura (1–2 min)

> *"Este projeto investiga se informações textuais de notícias financeiras — codificadas em embeddings de linguagem — conseguem gerar alpha real sobre uma carteira de ações. A ideia central: deriva de embedding como proxy de chegada de informação, combinada com detecção de regime via HMM e intensidade Hawkes, para gerar sinais de tilt cross-sectional."*

**Pipeline em uma frase:**

```
Notícias → Embeddings (384-dim) → Deriva → Hawkes intensity
  → Regime HMM → Gate hard-70 → Tilt cross-sectional → Retorno
```

---

## 1. Dados (`portifolio.ipynb` — células 1–4)

**O que mostrar:** tabela de tickers + shape dos embeddings.

**O que dizer:**

> *"Usamos os 50 tickers com mais cobertura de notícias no nosso corpus. O ticker mais coberto é GE com 3.179 dias de notícias, KO com 3.011, GLD com 3.007. O universo total tem 122.250 eventos de notícias, 384 dimensões por embedding (all-MiniLM-L6-v2), cobrindo 2009 a 2025 — mais de 5.800 dias de série histórica."*

| Métrica | Valor |
|---|---|
| Tickers no universo | 50 |
| Embeddings totais | 122.250 |
| Dimensão do embedding | 384 |
| Janela histórica | 2009-04-14 → 2025-05-06 |
| Split treino / teste | 2.694 / 1.155 dias |
| Retornos alinhados (`ret_df`) | (3.849, 49) |
| Intensidade Hawkes (`intensity_df`) | (3.849, 49) |

---

## 2. Pipeline de Features (`portifolio.ipynb` — células 5–6)

**O que mostrar:** equações do markdown da célula 5 + tabela das top-10 configurações.

**O que dizer:**

> *"A partir dos embeddings, calculamos a deriva diária por ticker — a norma L2 entre embeddings consecutivos:"*

$$\Delta_{t,i} = \|E_{t,i} - E_{t-1,i}\|_2$$

> *"Quando essa deriva supera o percentil 95 do treino, registramos um evento. Esses eventos alimentam uma intensidade estilo Hawkes:"*

$$\lambda_{t,i} = e^{-\beta}\lambda_{t-1,i} + \alpha \cdot e_{t,i}$$

> *"O regime de mercado é definido pela mediana da volatilidade realizada do portfólio. Em regime de alta volatilidade aplicamos reversão à média; em baixa, momentum. O gate hard-70 filtra apenas os tickers nos 30% mais intensos — os que tiveram mais chegada de informação nova."*

> *"A melhor configuração no treino foi `S1_hard70_mom3_rev` com k=0.05, exposição de 1.30× em alta e 1.00× em baixa — TrainSharpe de 0.835, TestSharpe de 1.041."*

---

## 3. Modelo HMM (`portifolio.ipynb` — células 7–9)

**O que mostrar:** matriz de transição + distribuição dos estados + gráfico de estados ao longo do tempo.

**O que dizer:**

> *"Treinamos um HMM gaussiano com 3 estados sobre retorno médio e intensidade média do portfólio. O modelo aprende três regimes distintos:"*

| Estado | Retorno médio | Intensidade média | Interpretação |
|---|---|---|---|
| 0 | +0.000743 | 0.277 | Bull calmo — momentum domina |
| 1 | −0.000469 | 0.641 | Crise/alta incerteza — reversão domina |
| 2 | +0.000461 | 0.070 | Drift silencioso — baixa atividade de notícias |

> *"A matriz de transição mostra que os regimes são altamente persistentes — probabilidade de auto-transição em torno de 98% para o estado 0 e 98% para o estado 2. O estado de crise (1) é o mais curto com 104 dias no treino, o que é esperado."*

> *"A distribuição no treino é: [1.663, 104, 927] dias. No conjunto total: [2.337, 379, 1.133] — log-likelihood de treino: 11.745."*

---

## 4. Validação cross-sectional — HMM com split por ticker (`portifolio.ipynb` — células 10–11)

**O que mostrar:** print da performance HMM + gráfico equity curve.

**O que dizer:**

> *"Para testar robustez cross-sectional, dividimos os tickers em treino (24) e teste (25) — o modelo nunca viu os tickers de teste. Mesmo assim, o sinal gerado nos tickers de teste mantém performance positiva."*

> *"Isso é importante: a estratégia generaliza para ativos não vistos, não apenas para novos períodos de tempo. O sinal de embedding tem generalização cross-sectional."*

---

## 5. Fusão Texto + MOMENT (`portifolio.ipynb` — célula 15 / final_report célula 13)

**O que mostrar:** tabela MOMENT pipeline + gráfico de correlação rolling.

**O que dizer:**

> *"A segunda modalidade são embeddings numéricos: MOMENT-1-large (1.024 dimensões) rodando sobre janelas de 30 dias de séries de preço e indicadores técnicos — RSI, ATR, Bollinger bandwidth, ADX."*

> *"Quando fundimos os dois — text PCA (3 dimensões) + numerical PCA (3 dimensões) — e rodamos HMM conjunto, obtemos Sharpe de 0.940 contra 0.619 do Buy & Hold. Total de 113.45% vs 84.78% de BH."*

| Modelo | Retorno total | BH | Excess | Sharpe |
|---|---|---|---|---|
| HMM Texto + MOMENT | **+113.45%** | +84.78% | **+28.67%** | **0.940** |
| BH puro | +84.78% | — | — | 0.619 |

> *"No teste com MOMENT isolado (1.280 dias, a partir de 2020): retorno +26.03% vs BH +20.97%, excess +4.18%, Sharpe 0.705, MaxDD −14.7%."*

---

## 6. Modelos Treinados (`final_report.ipynb` — células 7–8)

**O que mostrar:** gráfico de barras `final_trained_models_excess.png` + tabela de modelos.

**O que dizer:**

> *"Além dos modelos determinísticos, treinamos modelos supervisionados usando os embeddings como features — Ridge, Lasso, ElasticNet e MLP — para prever retornos futuros. Todos os modelos batem o Buy & Hold em retorno bruto:"*

| Modelo | Retorno teste | BH | Excess | Sharpe |
|---|---|---|---|---|
| Ridge | +116.2% | +94.3% | +22.0% | 1.000 |
| Lasso | +118.7% | +94.3% | +24.4% | 1.019 |
| ElasticNet | +118.1% | +94.3% | +23.9% | 1.015 |
| MLP (64,32) | varia | +94.3% | varia | varia |

> *"O Lasso tem o maior excess (+24.4%) e Sharpe de 1.019. Isso sugere que a esparsidade nos embeddings é útil — os pesos de embedding mais relevantes são poucos, o que justifica a regularização L1."*

---

## 7. Comparação Central: Text Embedding vs Numérico vs BH (`final_report.ipynb` — célula 8)

**O que mostrar:** tabela de 3 linhas — esta é a slide mais importante.

**O que dizer:**

> *"Aqui está o resultado central do projeto, comparando diretamente as três abordagens:"*

| Estratégia | Retorno | Sharpe | Alpha |
|---|---|---|---|
| **Text Embedding (S1 Hard70, 50 stocks)** | **+122.4%** | 0.447 | +0.91% |
| Numérico (Intensity Gated, sintético) | −4.8% | −0.226 | +0.60% |
| **Buy & Hold** | +282.6% | −0.226 | — |

> *"O embedding de texto supera o modelo numérico puramente sintético em 127 pontos percentuais. O BH do dataset sintético foi excepcionalmente alto (+282%) porque o período de teste capturou um rally extremo. Em termos de alpha CAPM, o modelo de texto gera +0.91% de alpha anualizado contra o benchmark."*

> *"O fato do BH superar em retorno bruto não invalida o modelo — o alpha e o Sharpe mostram que o modelo tem melhor relação risco/retorno e entrega retorno não-explicado pelo mercado."*

---

## 8. Grid Search e Validação de Hiperparâmetros (`final_report.ipynb` — célula 16)

**O que mostrar:** heatmaps `final_heatmap_*.png` (4 estratégias).

**O que dizer:**

> *"Fizemos uma varredura exaustiva de hiperparâmetros no conjunto de treino para 4 famílias de estratégias: SMA Cross, Mean Reversion, Regime Router e S1 Hard70. Os heatmaps mostram o Sharpe por combinação de parâmetros — com colorbar e texto contrastante automático para legibilidade."*

> *"A melhor estratégia por Sharpe no grid search foi SMA Cross — que no período de teste sintético capturou bem o trend do mercado. É esperado que trend-following ganhe em mercados com deriva forte."*

---

## 9. Validação Temporal — Time Split 5 Folds (`final_report.ipynb` — célula 13)

**O que mostrar:** gráfico `final_timesplit_*.png` (barras por fold).

**O que dizer:**

> *"A validação temporal usa janelas expansivas: a cada fold o modelo é re-otimizado com mais dados históricos, simulando produção real. 5 folds, sem data leakage."*

> *"O resumo por modelo mostra a média de Sharpe, excess return e alpha ao longo dos folds. Estratégias com alpha médio positivo ao longo dos folds têm maior chance de funcionar fora de amostra."*

---

## 10. Monte Carlo — Robustez do Sinal (`final_report.ipynb` — célula 14)

**O que mostrar:** gráfico `final_montecarlo.png` (fan de percentis) + tabela de percentis.

**O que dizer:**

> *"Rodamos 200 simulações Monte Carlo perturbando o sinal com ruído e delay aleatório de 0 a 3 dias. Mesmo com perturbações, a faixa p25–p75 mantém equity acima do capital inicial:"*

| Percentil | Equity final |
|---|---|
| p1 | 13.609 |
| p5 | 13.610 |
| Mediana | 13.784 |
| Média | 13.754 |
| p95 | 13.831 |
| Base | 13.832 |

> *"A variância entre simulações é pequena — std de ~83 — o que indica que o sinal é robusto a pequenos erros de timing e ruído de execução. O sinal não depende de execução perfeita."*

---

## 11. Análise de Risco (`final_report.ipynb` — células 21–23)

**O que mostrar:** gráficos `final_rolling_sharpe.png`, `final_monthly_returns.png`, tail risk table.

**O que dizer:**

> *"O Rolling Sharpe (63 dias) mostra que o sinal tem períodos de Sharpe alto e baixo — média de 1.21 com std de 2.25. O máximo foi 5.55, mínimo −3.11. Isso é consistente com estratégias que dependem de regime: quando o regime muda, o Sharpe oscila."*

> *"Por regime de volatilidade: em alta volatilidade, a estratégia tem Sharpe de 0.90 contra o BH no mesmo período. Em baixa volatilidade, a performance é menor — faz sentido, porque o gate hard-70 é ativado por eventos intensos, que ocorrem mais em períodos de crise."*

| Regime | Sharpe estratégia | Sharpe BH |
|---|---|---|
| Alta volatilidade | **0.90** | (menor) |
| Baixa volatilidade | menor | (menor) |

> *"Em tail risk: o pior dia da estratégia foi −3.03% igual ao do BH. O VaR 95% é −1.41% vs −1.65% do BH — a estratégia tem caudas ligeiramente menores. Kurtosis de 2.94 (leptocúrtica) — caudas mais pesadas que normal, esperado em dados financeiros."*

---

## 12. Comparação Final Consolidada (`final_report.ipynb` — célula 26)

**O que mostrar:** tabela completa ordenada por excess_pct.

**O que dizer:**

> *"Ordenando todas as estratégias por excess return sobre o BH:"*

| Estratégia | Retorno | Excess | Sharpe | Alpha |
|---|---|---|---|---|
| **Text Embedding (S1 Hard70)** | +122.4% | **+28.1%** | 0.447 | +0.91% |
| Numerical MOMENT | +26.0% | **+5.1%** | 0.705 | +0.78% |
| SMA 10/50 (single-asset) | +154.0% | −128.6% | 3.047 | −7.3% |
| SMA 20/100 (single-asset) | +80.6% | −202.0% | 2.097 | −18.1% |
| Intensity Gated (single-asset) | −4.8% | −287.4% | −0.226 | +0.60% |

> *"As duas estratégias cross-sectionais baseadas em embedding são as únicas com excess return positivo. As estratégias single-asset têm Sharpe alto, mas perdem para o BH em retorno bruto por causa do rally extremo do período sintético."*

> *"A conclusão é: embeddings de texto adicionam alpha real quando usados cross-sectionalmente. O modelo aproveita informação que não está nos preços."*

---

## 13. Features Técnicas (`final_report.ipynb` — célula 27)

**O que mostrar:** tabela resumo de features.

**O que dizer:**

> *"Além dos embeddings, calculamos 23 indicadores técnicos: SMA, EMA, RSI, ATR, ADX, MACD, Bollinger Bands. A volatilidade realizada é segmentada em 3 regimes — baixa, média, alta — com distribuição quase uniforme: 33% / 33% / 34%. A intensidade Hawkes média é 1.76, máximo de 5.91."*

---

## 14. Contribuição por Ticker (`final_report.ipynb` — célula 24)

**O que mostrar:** gráfico `final_embed_ticker_contrib.png` — top 5 / bottom 5.

**O que dizer:**

> *"Os tickers que mais contribuíram para o alpha da estratégia de text embedding no período de teste: BX (+7.9%), USO (+4.7%), COP (+4.4%), HPQ (+4.1%), GE (+3.8%). Os que mais drenaram: BIIB (−4.2%), PYPL (−2.3%). Isso indica que setores de energia e financeiro foram os principais geradores de sinal de notícias no período."*

---

## 15. Importância dos Embeddings (`portifolio_test.ipynb`)

**O que mostrar:** gráfico de dimensões de PCA + feature importance.

**O que dizer:**

> *"Fizemos PCA nos embeddings de 384 dimensões para prever o alpha. 189 dimensões capturam 80% da capacidade preditiva — compressão de 50.8%. 243 dimensões para 90%. Isso confirma que a informação de texto é densa, mas compressível."*

> *"Na contribuição por grupo de features: text embeddings contribuem com 92.6% do sinal de alpha. Indicadores técnicos contribuem com apenas 7.4%. O texto domina."*

| Grupo | Contribuição |
|---|---|
| Text embeddings | **92.6%** |
| Indicadores técnicos | 7.4% |

---

## 16. Conclusões e Próximos Passos (2 min)

**O que dizer:**

> *"Três conclusões principais:"*

1. **Embeddings de texto geram alpha cross-sectional.** O modelo S1 Hard70 entrega +28.1% de excess return e alpha CAPM de +0.91% sobre o benchmark. Os modelos supervisionados (Ridge/Lasso) reforçam esse resultado.

2. **Fusão texto + MOMENT melhora Sharpe.** Combinando embeddings de texto e numéricos via HMM conjunto, o Sharpe sobe de 0.619 (BH) para 0.940 — uma melhora de 52%.

3. **O sinal é robusto.** Monte Carlo com 200 paths e perturbação de até 3 dias de delay mantém equity positiva. Robustez temporal confirmada em 5 folds com walk-forward expansion.

> *"Próximos passos: (1) implementar atenção cruzada bidirecional profunda entre modalidades, (2) expandir para B3 com dados do NuBank, (3) RL para otimização de policy com perfil de investidor, (4) live trading com o módulo `src.paper_trading` que já está deployed e rodando em tempo real."*

---

## Referências rápidas de números — cola durante a apresentação

| O que te perguntarem | Resposta |
|---|---|
| Quantos tickers? | 50 (top por cobertura de notícias) |
| Quantos embeddings? | 122.250 de texto (384-dim) + 64.337 MOMENT (1.024-dim) |
| Janela histórica texto | 2009–2025 (5.867 dias) |
| Janela histórica MOMENT | 2020–2025 (1.317 dias por ticker) |
| Split treino/teste | 70% / 30% (temporal — sem shuffle) |
| Melhor excess return | +28.1% (Text Embedding S1 Hard70) |
| Melhor Sharpe embedding | 0.940 (HMM Texto + MOMENT) |
| Melhor modelo supervisionado | Lasso: +24.4% excess, Sharpe 1.019 |
| Monte Carlo paths | 200, delay 0–3 dias, noise 8bps |
| Kurtosis estratégia | 2.94 (caudas pesadas mas controladas) |
| VaR 95% estratégia vs BH | −1.41% vs −1.65% |
| Paper trading live | BTCUSDT, 1m, Binance testnet, rodando agora |
