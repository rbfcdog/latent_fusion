# Portfólio e Embeddings

## 1. O problema clássico de portfólio

Você quer escolher pesos $w$ para ativos tal que:

$$\max_w \; \mathbb{E}[R_p] - \lambda \, \mathrm{Var}(R_p)$$

onde:

$$R_p = w^\top r,\quad \mu = \mathbb{E}[r],\quad \Sigma = \mathrm{Cov}(r)$$

Tudo depende de:

- retornos esperados ($\mu$)
- covariância ($\Sigma$)

## 2. Markowitz portfolio theory

Formulação:

$$\max_w \; w^\top \mu - \frac{\lambda}{2} w^\top \Sigma w$$

ou:

$$\min_w \; w^\top \Sigma w \quad \text{s.t. } w^\top \mu = \mu^*$$

Problema prático:

- $\mu$ é muito difícil de estimar
- pequeno erro -> grande mudança no portfólio
- embeddings podem ajudar aqui

Como usar embeddings no Markowitz:

**Opção A — $\mu$ via embeddings**

- $\mu_i = f(e^{text}_i, features_i)$
- modelo prevê retorno esperado; entra direto no Markowitz

**Opção B — clusters semânticos como regularização**

- agrupar ativos por embeddings
- penalizar concentração:

$$\text{penalty} = \sum_{c \in \text{clusters}} \left(\sum_{i \in c} w_i\right)^2$$

- evita concentração em ativos semanticamente semelhantes

## 3. Black-Litterman

Ideia: mistura mercado (prior) e opiniões (views).

Prior:

$$\mu \sim \mathcal{N}(\pi, \tau \Sigma)$$

Views:

$$P\mu = q + \epsilon$$

Posterior:

$$\mu^* = [(\tau \Sigma)^{-1} + P^\top \Omega^{-1} P]^{-1}[(\tau \Sigma)^{-1}\pi + P^\top \Omega^{-1} q]$$

Tradução:

- $\pi$: retorno implícito do mercado
- $q$: suas previsões
- $P$: quais ativos você tem opinião
- $\Omega$: incerteza dessas opiniões

Onde entram embeddings:

- embeddings geram views

Exemplo:

- $pred\_return_i = f(e^{text}_i)$
- $confidence_i = g(e^{text}_i)$

Então:

- $q = pred\_return$
- $\Omega = \mathrm{diag}(1/\text{confidence})$

Interpretação:

- notícia forte -> alta confiança -> peso maior
- notícia fraca -> ignorada

Isso resolve:

- embeddings são ruidosos -> BL trata isso via incerteza

## 4. Risk parity

Ideia: não usa $\mu$, só risco:

$$w_i \propto \frac{1}{\sigma_i}$$

ou: cada ativo contribui igualmente para o risco.

Como usar embeddings:

- ajustar risco percebido

$$\sigma_{i,adj} = \sigma_i (1 + \text{semantic\_risk}_i)$$

Onde:

$$\text{semantic\_risk} = \frac{\text{volatilidade do embedding}}{\text{choque de notícia}}$$

Ativos com muita incerteza textual -> menos peso.

## 5. Estratégias modernas (mais próximas do seu projeto)

**A. Portfólio como função do embedding**

- direto: $w_t = f(e^{text}_t, x^{ts}_t)$
- treina com Sharpe ratio e retorno acumulado
- isso é end-to-end

**B. Alpha -> Portfólio**

- pipeline: embeddings -> alpha signal -> optimizer
- você já tem: HMM (beta) e embeddings (alpha)
- então: $\mu = alpha\_prediction$
- entra no Markowitz / BL

## 6. Como comparar métodos

Métricas:

- retorno médio
- volatilidade
- Sharpe ratio
- max drawdown

Comparações:

| Método | Descrição |
| --- | --- |
| Equal weight | baseline |
| Markowitz ($\mu$ histórico) | clássico |
| Markowitz + embeddings | seu |
| Black-Litterman | robusto |
| Black-Litterman + embeddings | principal |
| Risk parity | baseline robusto |

## 7. Insight forte para paper

"Embeddings são mais eficazes como geradores de views probabilísticas em frameworks bayesianos (Black-Litterman) do que como previsores diretos de retorno."

## 8. Conexão com o que você já fez

Você já tem:

- HMM -> regimes -> beta
- embeddings -> sinais fracos

Agora:

- embeddings -> views condicionais ao regime

$$q_t = f(e^{text}_t \mid S_t)$$

## 9. Melhor setup recomendado

Pipeline:

1. HMM -> regime
2. Embeddings -> alpha + confidence
3. Black-Litterman -> $\mu^*$
4. Markowitz -> pesos

## Final takeaway

- Markowitz precisa de $\mu$ (difícil)
- Black-Litterman é perfeito para embeddings (incerteza)
- Risk parity é baseline robusto
- embeddings funcionam melhor como geradores de opinião com incerteza do que como previsão direta
Markowitz → precisa de μ (difícil)
Black-Litterman → perfeito pra embeddings (incerteza!)
Risk parity → baseline robusto

👉 Embeddings são melhores como:

geradores de opinião com incerteza, não como previsão direta