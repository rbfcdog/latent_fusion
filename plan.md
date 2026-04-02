# Análises para Capturar Mudanças de Regime e Interação Texto × Série Temporal

## 1. Métricas para Captura de Mudança de Regime

### 1.1 Regime baseado em Volatilidade
- Calcular:
  - Rolling standard deviation dos retornos
  - OU modelo GARCH
- Aplicar clustering (ex: K-means ou HMM)
- Definir regimes:
  - Baixa volatilidade
  - Média volatilidade
  - Alta volatilidade

**Métricas:**
- Accuracy de classificação de regime
- F1-score por regime
- Matriz de confusão
- Previsão de transições de regime

---

### 1.2 Hidden Markov Model (HMM)
- Treinar HMM sobre os retornos
- Usar estados ocultos como proxy de regime

**Métricas:**
- Accuracy de previsão de estado
- F1-score
- Regime Detection Delay:
  - Diferença temporal entre mudança real e prevista

---

### 1.3 Change Point Detection
- Métodos:
  - Bayesian Change Point Detection
  - Biblioteca `ruptures`

**Métricas:**
- Precision / Recall na detecção de pontos de mudança
- Erro temporal médio (distance error)

---

## 2. Avaliação do Impacto do Texto

### 2.1 Ablation Study (essencial)
Comparar:
1. Modelo apenas com séries temporais
2. Modelo apenas com texto
3. Modelo multimodal

**Métricas:**
- RMSE / MAE (regressão)
- Accuracy / AUC (classificação)
- Sharpe Ratio (se houver backtest)

**Extensão importante:**
- Avaliar separadamente por regime

---

### 2.2 Conditional Performance Gain
Separar dados em:
- Com notícia relevante
- Sem notícia

**Métrica:**

Δ performance = erro_baseline - erro_multimodal


---

### 2.3 Lead-Lag Analysis
Objetivo: verificar se texto antecipa movimento de preço

**Procedimento:**
- Calcular cross-correlation entre:
  - embeddings textuais
  - retornos futuros

**Resultado:**
- Identificação do lag ótimo (ex: texto impacta preço em +2 dias)

---

## 3. Métricas de Fusão Texto × Série Temporal

### 3.1 Análise de Atenção Cruzada

#### Entropia da Atenção
- Baixa entropia → atenção focada
- Alta entropia → dispersão / ruído

Comparar entre:
- Regimes
- Eventos relevantes vs normais

---

#### Variação Temporal da Atenção

||Attn_t - Attn_{t-1}||


Interpretação:
- Grandes variações → possível mudança de regime

---

### 3.2 Mutual Information
- Medir dependência entre:
  - embeddings textuais
  - embeddings temporais

**Método:**
- MINE (Mutual Information Neural Estimation)

**Métrica:**

MI(texto, tempo)


Hipótese:
- Maior MI em regimes de alta volatilidade

---

### 3.3 Causal Impact
- Remover ou embaralhar texto

**Métrica:**

Causal Effect = |prediction_with_text - prediction_without|


---

### 3.4 Counterfactuals
- Substituir notícias:
  - positiva → negativa

**Analisar:**
- Variação na previsão

---

## 4. Métricas Financeiras (Avaliação Prática)

### 4.1 Backtest
Estratégia simples:
- Compra se retorno previsto > 0
- Venda se < 0

**Métricas:**
- Sharpe Ratio
- Max Drawdown
- Win Rate

---

### 4.2 Performance por Regime
- Avaliar desempenho do modelo em:
  - Alta volatilidade
  - Baixa volatilidade

---

## 5. Experimentos Avançados (Diferenciais)

### 5.1 Regime-aware Attention
- Clusterizar padrões de atenção
- Associar clusters a regimes de mercado

---

### 5.2 Text Shock Sensitivity
Definir:

shock = evento textual com alto impacto semântico


Medir:
- Variação da previsão após o evento

---

### 5.3 Robustez a Drift Temporal
- Treinar em período A (ex: 2020–2022)
- Testar em período B (ex: 2023–2024)

Comparar:
- Multimodal vs unimodal

---

### 5.4 Métrica de Alinhamento Latente (proposta nova)

Alignment Score = similarity(latent_text, latent_time)


- Usar cosine similarity

**Objetivo:**
- Medir qualidade do alinhamento entre modalidades ao longo do tempo

---

## 6. Estrutura Recomendada de Experimentos

1. Benchmark multimodal vs unimodal
2. Avaliação por regime
3. Lead-lag entre texto e preço
4. Análise de atenção
5. Mutual information
6. Teste de drift temporal