# Arquitetura do Transformer para Séries Temporais (Latent Fusion)

## 1. Visão Geral
O módulo `src/models/transformer_ts.py` implementa uma arquitetura **Encoder-Decoder Transformer** pura (nativa do PyTorch), adaptada especificamente para modelagem e previsão de séries temporais financeiras multivariadas no projeto Latent Fusion. 

Ele é projetado para receber múltiplos *features* em paralelo (ex: preços, *features* de microestrutura cripto, intensidade de Hawkes e embeddings textuais do FinBERT) e prever os próximos passos da série de forma autorregressiva.

---

## 2. Componentes Principais da Arquitetura

### 2.1 Positional Encoding Aprendível (`PositionalEncoding`)
Ao contrário do *Positional Encoding* senoidal clássico do paper *Attention Is All You Need*, esta arquitetura utiliza **parâmetros aprendíveis** (`nn.Parameter`).
- **Motivação:** Em séries temporais financeiras, a posição absoluta (ex: sazonalidade fixa) muitas vezes importa menos do que a posição relativa e a estrutura latente dos dados. Permitir que o modelo aprenda a codificação posicional dá mais flexibilidade.
- **Implementação:** Inicializado com uma distribuição normal (`torch.randn`) escalonada por um fator pequeno (`scale = 0.04`).

### 2.2 Encoder (`nn.TransformerEncoder`)
O Encoder processa a janela histórica de observações (`src`).
- **Input:** Matriz 3D `(Batch, seq_len_src, input_dim)`.
- **Mecanismo:** Usa *Self-Attention* para mapear dependências complexas no histórico recente do ativo, extraindo representações ricas das *features* (ex: como um pico de *Order Flow Imbalance* se relaciona com uma mudança no *Funding Rate*).

### 2.3 Decoder (`nn.TransformerDecoder`)
O Decoder é responsável por gerar as previsões futuras (`tgt`).
- **Masking:** Utiliza uma máscara triangular (`generate_square_subsequent_mask`) que impede o modelo de "olhar para o futuro" durante o treino.
- **Cross-Attention:** Faz a fusão entre o que foi gerado até o momento (o *target* mascarado) e a "memória" (output) extraída pelo Encoder.
- **Bug Corrigido:** Na versão depreciada do código, havia um vazamento/erro onde o `tgt` cru era passado ao decoder sem passar pelo *Embedding* e *Positional Encoding*. Isso foi corrigido passando `tgt_emb`.

### 2.4 Camada de Saída (`fc_out`)
Uma camada linear simples que projeta a dimensão oculta do modelo (`d_model`) de volta para a dimensão original dos dados (`input_dim`), permitindo prever o próximo vetor multivariado da série.

---

## 3. Dinâmica de Treinamento e Inferência

### 3.1 Treinamento com *Teacher Forcing*
Durante a função `train_transformer`, a arquitetura utiliza a técnica de **Teacher Forcing** (Forçamento de Professor):
- A entrada do Decoder (`tgt_input`) é a sequência de alvo original deslocada um passo para o passado (`tgt[:, :-1, :]`).
- A saída esperada (`tgt_output`) é a sequência deslocada para o futuro (`tgt[:, 1:, :]`).
- **Vantagem:** Isso torna o treinamento extremamente rápido e estável, pois o modelo sempre recebe o valor "real" do passo anterior ao prever o próximo, em vez de depender da sua própria previsão imperfeita.

### 3.2 Otimização
- **Loss:** `MSELoss` (Erro Quadrático Médio), padrão para regressão contínua.
- **Otimizador:** `Adam` com *Learning Rate* inicial padrão de `1e-3`.
- **Scheduler:** `StepLR`, que reduz o *Learning Rate* pela metade (`gamma=0.5`) a cada 10 épocas, permitindo um ajuste fino nas últimas fases do treino.

### 3.3 Inferência Autorregressiva (`predict`)
Em modo de produção (inferência), o *Teacher Forcing* não pode ser usado pois não sabemos o futuro. 
A função `predict` implementa um loop **autorregressivo**:
1. Pega o último passo conhecido do histórico (`src[:, -1:, :]`).
2. Passa pelo Transformer para prever o passo $T+1$.
3. Concatena essa previsão à entrada do Decoder.
4. Usa a nova sequência para prever $T+2$, e assim por diante.
Isso permite gerar previsões de horizonte flexível (`steps=N`).

---

## 4. Integração no Pipeline *Latent Fusion*

O Transformer foi desenhado para atuar como o **motor preditivo central** caso a abordagem de *Reinforcement Learning* (PPO) ou os modelos lineares (Ridge, Lasso) não consigam capturar a alta não-linearidade do mercado:

1. **Fusão de Sinais:** Ele recebe as matrizes numéricas geradas por `add_microstructure_features` (do `crypto_microstructure.py`) mescladas com os vetores do `sentiment.py`.
2. **Geração de Target:** A saída prevista (ex: retorno no passo $T+1$) pode ser transformada em um sinal probabilístico $\in [-1, 1]$.
3. **Execução:** Esse sinal é então ingerido pelo `BacktestEngine` ou serve de observação de estado (`obs`) para os agentes de RL em `rl_env.py`.

## 5. Próximos Passos Recomendados
- **Time2Vec:** Avaliar a substituição do *Positional Encoding* simples por técnicas baseadas em tempo contínuo (ex: Time2Vec), que lidam melhor com feriados e finais de semana (especialmente em dados B3/NASDAQ).
- **Probabilistic Forecasting:** Alterar a camada final `fc_out` para prever os parâmetros de uma distribuição (Média e Variância) em vez de uma estimativa pontual (Point Estimate), melhorando o controle de risco da estratégia.