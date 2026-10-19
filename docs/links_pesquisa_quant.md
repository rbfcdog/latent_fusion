# Links de Pesquisa — Quant e IA

Links salvos e documentados em 2026-10-18, com relação ao projeto de IC
(latent fusion: embeddings de texto + séries temporais para alocação
condicionada a regime).

## Reels — Quant

### Rahul's Research Logs (@rahulsnotebook) — Competições Kaggle para quant

<https://www.instagram.com/reel/DbObJ0jPGwk/> (25/07/2026)

- Para entrar em quant finance, programação é habilidade não negociável;
  hackathons e comunidade ensinam a pensar como desenvolvedor quant.
- Recomenda competições Kaggle para aprender modelagem, backtesting e
  desenvolvimento de estratégias com dados reais (propõe 5 competições,
  enviadas por DM ao comentar "quant").
- Vídeo verificado: o único frame recuperável é o card de título
  ("5 kaggle competitions you can take part in to test your knowledge and
  coding skills (quant finance edition)"); os nomes das 5 competições estão
  no vídeo, que exige login para baixar. Apenas a Gridworld (abaixo) foi
  identificada via link separado.

### David Arias, CFA (@davidarias_cfa) — Fontes gratuitas de dados de opções e volatilidade

<https://www.instagram.com/reel/Db0nGewOUtI/> (09/08/2026)

Quatro fontes para chains de opções, IV e preços históricos sem pagar feed:

| Fonte | Cobertura | Observações |
|---|---|---|
| **OptionsDX** (optionsdx.com) | Chains EOD desde 2010; intraday SPX/VIX/SPY | Todos strikes/expiries com bid, ask, volume, gregos, IV e preço do subjacente no mesmo arquivo. IV vem das opções, vol realizada do subjacente — a diferença é o variance risk premium. Download fatia a fatia. |
| **DoltHub** (`post-no-preference/options`) | 2.098 símbolos, 2019–hoje, quase todo o mercado de opções dos EUA | Banco SQL versionado como git (requer CLI `dolt`). Schema só date/expiry/strike/call-put/bid-ask — sem gregos nem IV, é preciso calcular. |
| **Kaggle** (`dudesurfin/spy-options-eod-volatility-surface`) | SPY, 2010–2023, 14 arquivos parquet (~600 MB) | Gregos e IV já computados, calls e puts lado a lado. Fonte com que o autor treinou a rede de superfície de IV. Melhor ponto de partida para vol surface sem engenharia de dados (limitação: 1 ticker, até 2023). |
| **CBOE** (cboe.com) | Volume por símbolo (diário–anual), série completa da VIX | Chains completas só no DataShop (pagas). Serve para contexto e term structure. |

Vídeo verificado (16 s, sem narração, só slides com música): os frames
confirmam o caption. Detalhe do frame do DoltHub: o schema exibido da tabela
`option_chain` lista colunas `delta, gamma, theta, vega, rho, vol` — em
contradição com o caption ("sem gregos nem IV"); provavelmente as colunas
existem mas vêm nulas/derivadas. Vale conferir ao clonar o repo dolt.

Relevância IC: a competição de superfície de vol existe em
`features/implied_volatility_surface.py`; o dataset Kaggle SPY é o caminho
mais curto para treinar/atualizar modelos de superfície sem custo.

### Austin Green (@atraintrades) — "Can't beat the IV Walls"

<https://www.instagram.com/reel/DdTNtTLNSLU/> (15/09/2026)

- Vídeo verificado: é uma imagem estática (o "reel" é uma figura). Painel
  "90% Winrate Quant Strategy ATRAI", gráfico de candles com zonas de
  suporte/resistência, P&L acumulado +$76.119,25, trade mais lucrativo
  +$2.835 (MESH5). Nenhum conteúdo metodológico — material promocional de
  "Profitable Traders Club" (guia gratuito). Descartável para a IC.

### Dr. Alvaro Cintas (@drcintas) — Semanal de IA (Week 32)

<https://www.instagram.com/reel/Db0vxQexdJI/> (09/08/2026)

Vídeo não recuperável (bloqueado em todas as instâncias testadas), mas o
caption é a transcrição completa da narração de 60 s:

Resumo "tudo que aconteceu em IA na semana em 60 segundos":

1. **kimi-k3-in-c** (open source) — engine em C99 puro de 176 KB que
   faz streaming dos experts do Kimi K3 (2T de parâmetros) direto do
   disco; roda sem GPU.
2. **Qwen3.8-Max** (Alibaba) — 2,4T de parâmetros, 95B ativos, contexto
   de 1M de tokens, compreensão nativa de vídeo/imagem; modelo chinês
   melhor ranqueado.
3. **Muse Code** (Meta) — primeiro coding agent; roda no terminal, mantém
   agentes de background vivos na sessão, trabalha o repo em worktrees
   paralelos.
4. **Grok Imagine Image 2.0** (xAI) — edição por região apontada,
   remoção de fundo, 5 imagens de referência, smart resize; nº 2 mundial
   em edição de imagem.
5. **Hermes Skills Hub** (Nous Research) — 90k+ skills comunitários
   gratuitos para agentes, 200 categorias.
6. **Pocket TTS** (Kyutai) — modelo de voz de 100M de parâmetros, clona
   voz localmente, 6x mais rápido que tempo real, setup de um comando.

### Vincent (@vince.quant) — Signature Trading (rough path theory)

<https://www.instagram.com/reel/DdSBAFqOlg7/> (14/09/2026)

O reel mais denso tecnicamente e o mais diretamente ligado à IC. Narração
transcrita do vídeo (82 s) — confirma o caption e fecha com a tese do
reel: "antes de construir outro indicador, compute a signature; seu
indicador feito à mão é próximo de uma soma ponderada dos termos dela, e
os melhores pesos são a fórmula de Markowitz sobre esses termos". Frames
mostram: MA/RSI/Kalman como f(past path); aproximação
f(path) ≈ ℓ₁S₁ + ℓ₂S₂ + ⋯; ℓ* ∝ Σ⁻¹μ ("Markowitz no espaço da
signature"); rodapé credita Futter, Horvath & Wiese (2025), Quantitative
Finance, Imperial College · Oxford.

Conteúdo técnico (caption):

- Toda estratégia sistemática é função do caminho passado. A teoria de
  rough paths dá a descrição canônica do caminho — a **signature**
  (sequência de integrais iteradas): nível 1 = incrementos, nível 2 =
  áreas varridas (contra o tempo, o formato do trend; entre dois ativos,
  o lead-lag), níveis maiores = forma fina, decaindo como 1/n!.
- Hambly & Lyons (2010, Annals of Mathematics): a signature determina o
  caminho; teorema de aproximação universal → qualquer função contínua do
  caminho (cruzamentos de média móvel, RSI, Bollinger, até Kalman) é
  aproxímável arbitrariamente bem por combinação linear de termos da
  signature.
- **Futter, Horvath & Wiese (2025, Quantitative Finance 25(2), arXiv
  2308.15135)** — "Signature Trading: A Path-Dependent Extension of the
  Mean-Variance Framework with Exogenous Signals": regredindo uma
  estratégia MACD momentum (com sigmoid) no ETF TLT sobre a signature,
  recuperam-na com R² de 66% (ordem 1), 89% (ordem 3), 98% (ordem 11).
  Como a estratégia é linear na signature, o ótimo de média-variância é
  fechado: pesos ∝ inversa da covariância das atribuições de lucro ×
  atribuições, lidos da expected lead-lag signature — sem rede neural,
  sem gradiente. Ordem 0 = Markowitz exato; cada ordem adiciona
  dependência de caminho e controle de drawdown embutido. Exemplo:
  2 ativos + 1 sinal + tempo na ordem 2 = 21 termos.
- Limitações do paper: expected signature é ruidosa (robustez fica para
  trabalho futuro); covariância exige termos de ordem 6 para estratégia
  de ordem 2; sem modelar market impact; evidência empírica = fronteira
  com 3 ETFs + uma replicação de momentum.
- Bibliotecas: `signatory`, `esig`, `iisignature`.

Relevância IC: alternativa de baixo custo computacional ao
`models/temporal_fusion.py` — a signature substitui features manuais e a
atenção do MOMENT/embeddings como descrição do caminho; combina linear
com sinal exógeno (a drift de embeddings da IC é exatamente um "exogenous
signal" no formalismo do paper).

### Carousel não acessível

<https://www.instagram.com/p/DbyB-woFX1s/> — post removido ou restrito
(embed retorna "the post may have been removed"); sem caption nos
metadados. Recuperar manualmente no app, se ainda disponível para a
conta.

## Competição Kaggle — Predicting Sales Quantity in our Dynamic Gridworld

<https://www.kaggle.com/competitions/predicting-sales-quantity-in-our-dynamic-gridworld>

- Host: Lennarth Haupts · Community Prediction Competition · sem pontos
  nem medalhas.
- Período: 27/08/2023 – 27/09/2023 (encerrada). 80 inscritos, 25
  participantes ativos, 351 submissões.
- Tarefa: prever quantidade de vendas em um Gridworld 2D de 150×150
  casas, com múltiplas cidades e lojas por casa; features incluem
  atributos de cidade, níveis de anúncio e preços. Tema: feature
  engineering em paisagem espacialmente correlacionada.
- Métrica: RMSE entre quantidades previstas e reais no conjunto de teste.

## IA

### MoneyPrinterTurbo (harry0703)

<https://github.com/harry0703/MoneyPrinterTurbo/blob/main/README-en.md>

Gerador all-in-one de vídeos curtos por IA: a partir de um
tópico/palavra-chave gera roteiro, locução, footage (banco de imagens ou
geração por IA: Pexels, Pixabay, Coverr, Seedance, MiniMax H3 etc.),
legendas e trilha, exportando em 9:16 / 16:9 / 1:1. Fluxos via AI Agent,
WebUI (Gradio), API e CLI; publicação direta em TikTok, Instagram e
YouTube Shorts.

- Python 3.11+, sem GPU obrigatória; providers de LLM incluem OpenAI,
  Claude, Gemini, DeepSeek, Qwen, Kimi, Ollama etc.; TTS inclui Edge TTS
  (gratuito, sem chave).
- Relevância IC: ferramenta para produzir conteúdo (reels de divulgação
  do projeto/paper) — sem relação com o pipeline de pesquisa.

### PaperBanana (llmsresearch/paperbanana)

<https://github.com/llmsresearch/paperbanana>

Implementação open source (não oficial, MIT, Python 3.10+) do paper
"PaperBanana: Automating Academic Illustration for AI Scientists"
(arXiv 2601.23265, Google Research). Framework agêntico que gera figuras
acadêmicas com qualidade de publicação e plots estatísticos a partir de
texto:

- Pipeline multiagente de duas fases (até 7 agentes) com refinamento
  iterativo; providers OpenAI (GPT-5.2 + GPT-Image-1.5), Azure, Gemini
  (free tier) e Atlas Cloud.
- Input optimization, auto-refine com feedback, batch via manifest
  (YAML/JSON), inputs em PDF (PyMuPDF) para contexto de metodologia.
- CLI (`paperbanana generate`), API Python, servidor MCP e Studio
  (Gradio local); skills Claude Code para `/generate-diagram`,
  `/generate-plot`, `/evaluate-diagram`.
- Guias de estilo por venue (ACL, ICML, IEEE) para plots e metodologia;
  avaliação comparativa com VLM-judge.

Relevância IC: diretamente aplicável ao paper (latex/main.tex) — geração
de diagramas da arquitetura de fusão e plots estatísticos nos padrões
de venue, alternativa/aumento ao fluxo manual de matplotlib.

## Próximos passos (iteração com a IC)

1. **Signature trading (Futter et al. 2025)**: ler o arXiv 2308.15135 e
   avaliar `esig`/`iisignature` como baseline linear contra o
   TemporalFusion — a drift de embeddings vira sinal exógeno no
   formalismo.
2. **Dataset SPY options (Kaggle)**: baixar `dudesurfin/
   spy-options-eod-volatility-surface` para alimentar/atualizar
   `features/implied_volatility_surface.py`.
3. **PaperBanana**: usar para figuras do paper no `latex/`.
4. **MoneyPrinterTurbo**: apenas divulgação (reels), fora do escopo de
   pesquisa.
5. Recuperar manualmente o conteúdo do carousel `DbyB-woFX1s`.
