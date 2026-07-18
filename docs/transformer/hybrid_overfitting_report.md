# Relatório de Overfitting: Transformer em Carteira Híbrida

Este documento detalha os resultados dos testes rigorosos de *overfitting* aplicados ao modelo **TimeSeriesTransformer** (`src/models/transformer_ts.py`).

## 1. Contexto e Metodologia do Teste

O Transformer foi testado em uma **carteira híbrida** contendo criptomoedas, ações da NASDAQ e da B3, extraídas do dataset `combined_1d.parquet`.
- **Ativos válidos (>300 dias):** 27 *tickers*
- **Amostras temporais:** 557 dias
- **Arquitetura:** Transformer Encoder-Decoder (2 camadas cada, 4 *heads*, `d_model=64`)
- **Pipeline:** O modelo recebeu os últimos 20 dias (de 27 ativos em paralelo) para prever os retornos futuros. As alocações do portfólio foram feitas aplicando um `softmax` sobre as previsões do modelo (estratégia *long-only*).

---

## 2. Resultados dos Testes de Validação

Os testes foram desenhados para identificar memorização de ruído (característica comum de redes neurais profundas em finanças).

### Teste 1: Gap de Performance (Treino vs Teste)
| Métrica | Conjunto de Treino | Conjunto de Teste |
|---|---|---|
| **Retorno do Portfólio** | +82.595,59% | -11,25% |
| **Buy & Hold (Baseline)** | +4,15% | -8,26% |
| **Índice Sharpe** | 6.146 | -0.278 |
| **Erro (MSE)** | 0.5685 | 1.4446 |

**Veredito:** 🚨 **OVERFITTING EXTREMO.** O modelo memorizou a trajetória de preços do conjunto de treino (retorno astronômico), mas quebrou no conjunto de teste, não conseguindo sequer superar a estratégia *Buy & Hold*.

### Teste 2: Teste de Permutação (Aprendizado de Ruído)
Treinamos o modelo com os alvos (`targets`) **embaralhados aleatoriamente**. Se o modelo tem capacidade de memorizar ruído puro, ele terá alta performance no treino embaralhado.
- **Retorno no Treino Embaralhado:** +8.975,35% (Sharpe 4.325)
- **Retorno no Teste (Real):** -16,66% (Sharpe -0.580)

**Veredito:** 🚨 **FALHOU.** A rede possui capacidade em excesso. O *Transformer* decorou os ruídos puramente aleatórios do treino, confirmando que a arquitetura é grande demais para a quantidade de dados.

### Teste 3: Baseline Aleatório
Comparamos o desempenho do modelo no teste contra uma alocação com pesos totalmente aleatórios gerados a cada dia.
- **Transformer (Teste):** -11,25%
- **Portfólio Aleatório:** -7,10%

**Veredito:** 🚨 **FALHOU.** O modelo performou ligeiramente pior do que distribuir os pesos jogando dados.

---

## 3. Conclusão e Próximos Passos

O teste validou nossa infraestrutura de detecção de *overfitting*, que cumpriu seu papel impecavelmente ao soar o alarme. 

**O Diagnóstico:** 
A arquitetura do Transformer (`2 layers`, `64 d_model`) tem milhões de parâmetros. Aplicá-la em uma janela de apenas 557 dias para 27 ativos leva inevitavelmente ao colapso por "decoração" (*memorization*), conforme provado pelo Teste de Permutação.

**Ações Corretivas Necessárias para o Transformer:**
1. **Redução de Capacidade:** Diminuir `d_model` para 16 ou 32, e as camadas para 1.
2. **Aumento Drástico de Regularização:** Aumentar severamente o `dropout` (ex: 0.4) e aplicar *Weight Decay* (L2) agressivo no `AdamW`.
3. **Aumento do Dataset:** Séries financeiras diárias exigem de 10 a 20 anos de dados (*data augmentation* ou barras de 5 min) para alimentar um Transformer sem que ele tenha "fome de dados" e decore a amostra.
4. **Early Stopping:** Monitorar a *validation loss* ativamente durante o *Teacher Forcing* para interromper o treinamento na época 2 ou 3 (a *Train Loss* continuou caindo até 0.56, sinal claro de memorização excessiva).