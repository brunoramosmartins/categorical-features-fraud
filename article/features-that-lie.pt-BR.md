# Quando uma feature parece boa demais para ser verdade

### Fundamentos estatísticos de engenharia de features categóricas para detecção de fraude — da codificação à inferência

> **Nota:** esta é uma **tradução para revisão** (pt-BR). A versão canónica do artigo permanece em inglês: [`features-that-lie.md`](features-that-lie.md).

---

## Resumo

Em detecção de fraude, uma categoria com taxa observada de 100% de fraude em poucas transações é frequentemente tratada como um sinal quase perfeito. Uma matriz de correlação entre features construídas é frequentemente calculada — e depois deixada sem regra de decisão. Ambas as práticas confundem quantidades de **amostra** com verdades de **população**. Este artigo trata taxas alvo ao nível da categoria e muitas codificações como **estimadores estatísticos**: têm variância, modos de falha sob baixo suporte e protocolos de estimação que determinam se vazam informação do rótulo. Revisamos estimação por máxima verossimilhança para proporções binomiais, intervalos Agresti–Coull para $n$ pequeno e suavização Beta–Binomial (bayesiana) como média a posteriori conjugada. Unificamos codificações comuns perguntando que quantidade populacional cada uma estima, explicamos por que a correlação de Pearson entre features codificadas por target não justifica eliminar uma feature sem evidência orientada ao alvo, e ligamos vazamento de target ao cálculo desses estimadores na amostra errada. Quatro experimentos reproduzíveis num dataset sintético de fraude ilustram intervalos largos para categorias raras, suavização e generalização, a armadilha correlação versus redundância, e lacunas treino–teste sob codificação alvo com vazamento. A secção final oferece uma lente de decisão compacta: **todo valor codificado é uma estimativa — respeite o tamanho amostral e o âmbito de ajuste.**

**Palavras-chave:** features categóricas, target encoding, estimação binomial, suavização bayesiana, vazamento, detecção de fraude.

---

## 1. Introdução

Numa entrevista técnica para uma função em fraude, dois momentos ficaram comigo — não como falhas, mas como situações em que a análise ficou aquém da profundidade que o problema merecia.

**Primeiro:** uma feature de país mostrava 100% de taxa de fraude para o Uruguai. O instinto era confiar; a orientação do entrevistador foi removê-la. **Segundo:** produziu-se uma matriz de correlação entre features; observaram-se correlações altas, mas não havia enquadramento para o que fazer a seguir — qual feature retirar, com que princípio e com que evidência.

Esses dois momentos parecem diferentes à superfície. Por baixo, partilham o mesmo erro: tratar uma **estatística amostral** como se fosse um **parâmetro populacional** com incerteza desprezível. Uma taxa de 100% em cinco linhas não tem o mesmo peso probatório que uma taxa de 0,4% em quarenta mil. Uma correlação de Pearson elevada entre duas colunas **derivadas do target** não implica que uma coluna seja **redundante para prever** o rótulo.

A tese deste artigo é estatística, não categórica:

> *Uma categoria com taxa alvo de 100% e cinco observações é **mais provavelmente** evidência de **dados insuficientes** do que de um sinal preditivo robusto. Uma correlação elevada entre duas features codificadas é **mais provavelmente** evidência de **variação partilhada** do que de redundância na forma como informam o alvo. Ambas as ilusões vêm de ignorar que todo valor codificado usado em aprendizagem supervisionada é um **estimador** — com variância, condições de viés e uma amostra correcta em que deve ser ajustado.*

O eixo central é simples: **toda estatística ao nível da categoria usada em engenharia de features é um estimador.** O tamanho amostral $n_c$ no nível $c$ controla quanta confiança $\hat{p}_c = k_c/n_c$ merece. Suavização não é mera regularização — é a média a posteriori sob um modelo Beta–Binomial quando esse é o prior escolhido [3,4]. Vazamento não é um capítulo moral à parte — é o que acontece quando um estimador de $P(Y\mid X=c)$ é calculado com informação que não existirá em produção [6].

**Roteiro.** A secção 2 enquadra o EMV e a sua variância. A secção 3 desconstrói o padrão “Uruguai” com intervalos Agresti–Coull [1,2]. A secção 4 apresenta suavização Beta–Binomial como média ponderada de dados e prior. A secção 5 cataloga codificações pelo **estimando**. A secção 6 dá um contraexemplo formal correlação versus redundância. A secção 7 propõe uma heurística de decisão orientada ao alvo após calcular correlações. A secção 8 trata brevemente o vazamento de target. A secção 9 resume quatro experimentos (Figuras 1–4). A secção 10 sintetiza uma lente de decisão. A secção 11 conclui.

**Quem deve ler.** Modeladores de fraude e risco que colocam features categóricas em produção; cientistas de dados a preparar entrevistas onde aparecem “100% em cinco linhas” e “matrizes de correlação”; leitores que querem uma narrativa estatística **única** com código **reproduzível** em vez de um levantamento de bibliotecas.

---

## 2. Uma estatística ao nível da categoria é um estimador

Seja $X$ uma feature categórica e $c$ um nível fixo. Entre as $n_c$ linhas de treino com $X=c$, seja $k_c$ o número com $Y=1$ (fraude). A **proporção amostral**

$$
\hat{p}_c = \frac{k_c}{n_c}
$$

é o **estimador de máxima verossimilhança** (EMV) de $p_c = P(Y=1\mid X=c)$ sob um modelo binomial: condicionado a $X=c$, cada $Y$ é Bernoulli$(p_c)$ [7].

A variância amostral (condicionando em $n_c$) é

$$
\mathrm{Var}(\hat{p}_c) = \frac{p_c(1-p_c)}{n_c}.
$$

**Baixo suporte** significa $n_c$ pequeno, logo a variância é grande: $\hat{p}_c$ é um **estimador de alta variância no regime de baixo suporte**. O valor **observado** pode ser extremo mesmo quando $p_c$ é moderado.

Surge uma patologia separada para estimativas **plug-in** de variância $\hat{p}_c(1-\hat{p}_c)/n_c$: quando $\hat{p}_c\in\{0,1\}$, esta expressão é **zero**, sugerindo — falsamente — que não há incerteza. Métodos de intervalo para proporções binomiais (Wilson, Agresti–Coull, Clopper–Pearson) mantêm-se largos nesse regime [1,2].

**Consequência.** Target encoding que mapeia o nível $c$ para $\hat{p}_c$ não produz uma “verdadeira propensão à fraude” gravada na categoria; produz uma **estimativa** cuja fiabilidade é governada por $n_c$.

**Assimptótica e modelação.** Para $n_c$ grande, o EMV é aproximadamente normal com a variância acima, e intervalos de Wald tornam-se utilizáveis quando $\hat{p}_c$ não está no limite. Em contextos de fraude, **a maioria** dos níveis pode ter $n_c$ moderado enquanto uma **cauda longa** de níveis raros conduz a questões de política — precisamente onde aproximações normais e variância plug-in falham em simultâneo. O takeaway de engenharia não é “nunca usar $\hat{p}_c$”, mas **nunca confundir $\hat{p}_c$ com $p_c$** quando $n_c$ é pequeno.

---

## 3. O problema de baixo suporte: uma desconstrução numérica

Tome-se a âncora narrativa: **Uruguai**, com cerca de **cinco** transações, todas fraudulentas: $k_c=n_c=5$, logo $\hat{p}_c=1$.

A proporção ajustada Agresti–Coull usa pseudo-contagens:

$$
\tilde{p} = \frac{k+2}{n+4}, \qquad \tilde{n} = n+4.
$$

Para $(k,n)=(5,5)$, $\tilde{p}=7/9\approx 0.78$. Um intervalo aproximado a 95% usa $\tilde{p} \pm z_{0.975}\sqrt{\tilde{p}(1-\tilde{p})/\tilde{n}}$, produzindo uma faixa larga — frequentemente até cerca de **0,5** e **1** após truncagem [1]. O título “100% fraude” é **compatível** com um $p_c$ verdadeiro muito abaixo de 1.

Contraste com um país grande como o **Brasil**: se $n_c$ é da ordem de $10^4$ e $\hat{p}_c\approx 0{,}004$, a mesma maquinaria de intervalos produz uma faixa **estreita** (largura da ordem de $10^{-3}$). O estimador é informativo porque **$n_c$ é informativo**.

**Moral (suavizada).** A perfeição observada para o Uruguai é **mais plausivelmente** um artefacto de **$n$ pequeno** do que evidência de uma taxa populacional extrema **estável** — exactamente a formulação estatística que se quer numa sala onde “100%” soa decisivo.

| Âncora | $n_c$ (ilustrativo) | $k_c$ | $\hat{p}_c$ | Intervalo AC 95% (ordem de grandeza) |
|--------|---------------------|-------|-------------|--------------------------------------|
| Uruguai | $\approx 5$ | $\approx 5$ | $1.0$ | Largo (ex.: extremo inferior $\approx 0{,}5$) |
| Brasil | $\approx 10^4$ | $\approx 0{,}004\,n_c$ | $\approx 0{,}004$ | Estreito (largura $\sim 10^{-3}$) |

Os números na linha do Brasil são **ilustrativos** de um nível com $n$ alto perto da taxa base global; o teu dataset diferirá. A **estrutura** é o ponto: a mesma pergunta estatística (“qual é $p_c$?”) tem **precisão** radicalmente diferente entre níveis.

---

## 4. Suavização bayesiana: a solução fundamentada

Coloque-se um prior Beta em $p_c$:

$$
p_c \sim \mathrm{Beta}(\alpha,\beta),
$$

e verosimilhança $k_c \mid p_c \sim \mathrm{Binomial}(n_c, p_c)$. O posterior é conjugado:

$$
p_c \mid k_c, n_c \sim \mathrm{Beta}(\alpha + k_c,\; \beta + n_c - k_c),
$$

com **média a posteriori**

$$
\tilde{p}_c^{\mathrm{Bayes}} = \frac{\alpha + k_c}{\alpha + \beta + n_c}.
$$

Escrevendo $m=\alpha+\beta$ e média a priori $\mu_0=\alpha/(\alpha+\beta)$, então

$$
\tilde{p}_c^{\mathrm{Bayes}} = w_c \hat{p}_c + (1-w_c)\mu_0,
\qquad
w_c = \frac{n_c}{n_c + \alpha + \beta}.
$$

Quando $n_c$ é pequeno, $w_c$ é pequeno: a estimativa **encolhe** para $\mu_0$ (p.ex. a taxa global de fraude). Quando $n_c$ é grande, $w_c\to 1$: a média a posteriori **segue** o EMV [3,4].

**Interpretação.** Parâmetros de suavização em bibliotecas como `category_encoders` podem ser lidos como **força do prior** relativamente a $n_c$ — não como um botão arbitrário [5].

**Escolher $(\alpha,\beta)$.** Um default comum é fixar a média a priori $\mu_0=\alpha/(\alpha+\beta)$ na taxa global de fraude de **treino** $\bar{p}$, e seleccionar a massa total do prior $m=\alpha+\beta$ por validação cruzada ou orientação de domínio: $m$ maior puxa níveis raros com mais força para $\bar{p}$. Isto espelha prática empírico-bayesiana: o prior não é “crença” no sentido metafísico mas um **regularizador** cuja força se afina contra desempenho em hold-out — permanecendo interpretável como média a posteriori Beta [3,4].

**Ligação à produção.** Em scoring, linhas novas com um nível raro $c$ ainda usam contagens $(k_c,n_c)$ de **treino** (ou de uma janela móvel de treino). A suavização estabiliza o valor codificado enviado ao modelo quando $n_c$ é minúsculo; **não** elimina a necessidade de monitorizar o suporte de $c$ ao longo do tempo.

---

## 5. O panorama das codificações: o que cada uma estima?

| Codificação | Fórmula (nível $c$) | Estima | Usa $Y$? | Risco de vazamento | Alta cardinalidade |
|-------------|---------------------|--------|----------|-------------------|-------------------|
| One-hot | $\mathbb{1}[X=c]$ | Pertencer a $c$ | Não | Nenhum | Fraco (desenho largo e esparsa) |
| Frequência | $n_c/N$ | $\hat{P}(X=c)$ | Não | Nenhum | Bom (uma coluna) |
| Target (naïve) | $k_c/n_c$ | $P(Y{=}1\mid X{=}c)$ EMV | Sim | **Alto** se mal ajustado | Bom |
| Target suavizado | $(k_c{+}\alpha)/(n_c{+}\alpha{+}\beta)$ | Mesmo estimando, média posterior | Sim | Reduzido vs extremos; ainda mal ajuste se âmbito errado | Bom |

Boosters de árvores com suporte **nativo** a categóricas procuram divisões como $X\in S$ para subconjuntos $S$ de níveis; **não é obrigatória** codificação numérica manual para o learner usar a feature. Codificações tipo target podem ainda acrescentar um **resumo escalar** de $P(Y\mid X{=}c)$ junto das categorias brutas; se isso ajuda é **empírico**, e colunas derivadas do target devem ser ajustadas com **validação cruzada / cross-fitting** quando os rótulos não podem vazar entre folds [7].

**Codificação por frequência** estima prevalência $P(X{=}c)$, não $P(Y\mid X{=}c)$. **Não** transporta informação do rótulo e portanto **não** há vazamento de target pelo mapa de codificação em si. Ainda pode ajudar árvores e modelos lineares quando a **raridade** de um nível é preditiva de $Y$ mesmo que a taxa de fraude **naïve** nesse nível seja instável.

**Embeddings** (vectores densos aprendidos para níveis) estão fora do âmbito deste artigo: introduzem outro estimando e ciclo de optimização. A lente do estimador aplica-se se embeddings forem treinados com informação do rótulo — **âmbito de ajuste** e **vazamento** continuam centrais.

---

## 6. Correlação não é redundância

A segunda âncora narrativa: calculou-se uma **matriz de correlação**; observou-se correlação alta; a análise **parou** antes de uma decisão orientada ao alvo.

**Facto.** A correlação de Pearson entre duas colunas **codificadas por target** mede **associação linear entre linhas** entre as duas estimativas ao nível atribuídas a cada linha. **Não** implica que as duas features sejam **condicionalmente redundantes** para $Y$: $P(Y\mid X_1)$ e $P(Y\mid X_2)$ podem diferir fortemente para pares específicos de níveis mesmo quando as colunas codificadas correlacionam.

**Brinquedo trabalhado (oito linhas).** Níveis $X_1\in\{A,B,C\}$, $X_2\in\{M,N,P\}$, $Y$ binário:

| Linha | $X_1$ | $X_2$ | $Y$ |
|-------|-------|-------|-----|
| 1 | C | M | 0 |
| 2 | A | P | 1 |
| 3 | C | M | 0 |
| 4 | B | P | 1 |
| 5 | B | M | 0 |
| 6 | B | M | 1 |
| 7 | B | P | 0 |
| 8 | C | N | 0 |

Target encodings naïves de treino dão $\mathrm{Corr}(z_1,z_2)\approx 0{,}72$, mas $P(Y{=}1\mid X_1{=}A)=1$ enquanto $P(Y{=}1\mid X_2{=}M)=1/4$, e um modelo logístico com **ambas** as codificações pode superar qualquer uma sozinha nesta tabela. **Correlação marginal não é redundância condicional.**

**Experimentos em pipeline.** Num dataset sintético grande de fraude (ver §9), a correlação de Pearson entre target encodings **por linha** de `country` e `merchant_category` pode cair **abaixo** de $0{,}7$ porque muitos níveis **agregam** linhas e diluem a associação linear ao nível da linha. O **princípio** mantém-se: inspeccionar comportamento **condicional** e **métricas de modelo** ligadas a $Y$, não $|r|$ sozinho. A tabela-toy é o **padrão de prova**; o código é o **teste de stress**.

---

## 7. Quando retirar, quando manter

**Inseguro (para codificações derivadas do target).** Retirar uma feature porque $|r|>0{,}9$ com outra **sem** verificar a relação com $Y$.

**Triagens mais seguras.** Informação mútua com $Y$, importância por permutação, ou explicações de modelo — perguntam se a feature **move** a predição de $Y$, não só se acompanha outra feature [7].

**Heurística em cinco passos** (após calcular correlações):

1. Sinalizar $|r|$ alto entre colunas codificadas.
2. Para cada par sinalizado, pontuar **cada** feature contra $Y$ (p.ex. informação mútua ou importância por permutação).
3. Se **ambas** levam sinal, **manter ambas** salvo necessidade de modelo mais simples — documentar porquê.
4. Se uma é inerte, considerar retirar a inerte — documentar a métrica.
5. **Escrever a decisão** com números, não “removemos features colineares.”

Isto era o que faltava quando o caso parou em “vimos correlação alta.”

---

## 8. Vazamento de target: quando o estimador vê a resposta

**Vazamento** (sentido restrito aqui): o valor de uma feature para uma linha **depende do rótulo dessa linha** (ou de dados futuros) de forma que não existirá em tempo de scoring.

**Mecanismo (target encoding naïve).** Calcular $\hat{p}_c=k_c/n_c$ usando **todas** as linhas, incluindo a linha a codificar. A contagem $k_c$ **inclui** o $Y_i$ da linha actual, logo a codificação **codifica a chave de respostas** para essa linha. Métricas de treino incham; a comparação correcta é estatísticas **out-of-fold** ou leave-one-out ajustadas **sem** o rótulo da linha, ou âmbitos estritamente **só treino** [6].

**Baixo suporte amplifica a gravidade.** Quando $n_c$ é minúsculo, um rótulo move $\hat{p}_c$ fortemente — logo a auto-influência é grande.

**Teste de deteção rápido.** Se o AUC-PR de treino dispara enquanto o de validação mal se move após introduzir target encoding, **primeiro** verificar se as codificações foram ajustadas com informação de rótulo proibida.

---

## 9. Experimentos

**Reprodutibilidade.** A partir da raiz do repositório, com Python 3.10+:

```text
python -m venv .venv && source .venv/bin/activate   # ou Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_all.py
```

As figuras são gravadas em `figures/` ao DPI definido em `config.yaml` (default 300). O gerador sintético está documentado em `docs/dataset-design.md`; ressalvas e intervalos numéricos resumem-se em `docs/experiments-summary.md`.

### Experimento A — A ilusão da “feature perfeita”

**Configuração.** Divisão estratificada treino/teste; **todas** as linhas do Uruguai ficam em **treino** para a âncora de baixo suporte ter $n_c{=}5$, $k_c{=}5$ no treino. Taxas alvo naïves $\hat{p}_c$ por país e intervalos Agresti–Coull a 95% no split de treino. Um segundo painel varia a força do prior $m=\alpha+\beta$ para um Beta centrado na taxa global de fraude de treino.

**Figura 1** (`figures/exp_a_perfect_feature.png`). **Esquerda:** $\hat{p}_c$ versus $n_c$ (escala log) com barras de erro; Uruguai anotado. **Direita:** média a posteriori suavizada para o Uruguai versus $m$, com linha horizontal em $\bar{p}$.

**Observação.** O intervalo para o Uruguai permanece largo apesar de $\hat{p}_c{=}1$; aumentar $m$ puxa a estimativa suavizada para $\bar{p}$.

**Ligação teórica.** §§2–4: variância do EMV, Agresti–Coull, encolhimento Beta–Binomial.

### Experimento B — Suavização e generalização

**Configuração.** XGBoost com features numéricas mais **país** codificado de três formas: target naïve, target suavizado (hiperparâmetros de `config.yaml`), e one-hot. Métricas: AUC-PR treino/teste e F1 ao limiar $0{,}5$.

**Figura 2** (`figures/exp_b_smoothing_effect.png`). AUC-PR de teste por codificação; barras agrupadas treino vs teste AUC-PR.

**Observação.** Target naïve mostra frequentemente uma **maior** lacuna treino–teste do que o suavizado; a ordenação exacta varia com seed e hiperparâmetros.

**Ligação teórica.** §4: redução de variância por encolhimento; §5: o que cada codificação estima.

### Experimento C — Correlação versus redundância

**Configuração.** Target encodings naïves para `country` e `merchant_category` ajustados **só no treino**, aplicados a linhas treino/teste. Pearson $r$ nas linhas de treino; informação mútua com $Y$; XGBoost com numéricas + ambas as codificações versus retirar uma coluna.

**Figura 3** (`figures/exp_c_correlation_trap.png`). Dispersão das codificações; gráfico de barras AUC-PR de teste para {ambas, só país, só merchant}; barras de IM.

**Observação.** $|r|$ alto **não** autoriza retirar uma feature sem evidência orientada ao alvo; nalguns sorteios, uma codificação **aproxima** o sinal da outra — então o gráfico de barras reflecte **sobreposição**, não falha de lógica. A **regra de decisão** continua a ser a §7.

**Ligação teórica.** §§6–7.

### Experimento D — Vazamento via âmbito da codificação

**Configuração.** **Com vazamento:** target naïve para `country` com rótulos de **treino e teste concatenados**. **Correcto:** encoding out-of-fold nas linhas de treino; teste codificado só com estatísticas **só treino**. Mesmos hiperparâmetros XGBoost. O script pode usar uma fracção de teste **maior** para amplificar o efeito de rótulos agregados (ver `scripts/experiment_d_encoding_comparison.py`).

**Figura 4** (`figures/exp_d_encoding_comparison.png`). AUC-PR treino vs teste para pipelines com e sem vazamento.

**Observação.** AUC-PR de treino pode ser **optimista** com vazamento; o padrão de lacunas sustenta a §8.

**Ligação teórica.** §8; âmbito de ajuste do estimador [6].

---

## 10. Um enquadramento para decisões de codificação

Pense em três eixos: **cardinalidade** (quantos níveis), **suporte** ($n_c$ típico), e **família de modelo** (linear/NN versus árvore com categóricas nativas).

- **Alta cardinalidade, baixo suporte em muitos níveis:** preferir **target suavizado** ou **frequência** mais modelos regularizados; evitar interpretar $\hat{p}_c$ naïve com $n_c$ minúsculo como verdade absoluta.
- **Modelos lineares / NN:** precisam de entradas numéricas ou embedded; **one-hot** ou codificações **tipo target** são típicas; target-type exige **cross-fitting**.
- **Árvores gradient-boosted com categorias nativas:** codificação opcional; targets podem ainda acrescentar sinal escalar global — validar.
- **Depois de qualquer matriz de correlação sobre codificações:** correr a heurística da **§7** antes de retirar features.

A tabela abaixo é uma **bússola**, não lei: validar sempre em dados de hold-out.

| Cardinalidade | Suporte típico | Modelo (exemplos) | Codificação de primeira linha | Âmbito de ajuste para mapas baseados em target |
|---------------|----------------|-------------------|--------------------------------|-----------------------------------------------|
| Baixa | Alto por nível | Regressão logística | One-hot ou target (suavizado) | Só treino; CV para target |
| Alta | Cauda pesada de $n_c$ pequeno | XGBoost (cat nativo) | Nativo + opcional target suavizado | OOF / só treino para target |
| Alta | Cauda pesada | Rede neural | Embedding ou frequência + numérico | Embeddings: evitar vazamento de rótulo no grafo de treino |
| Qualquer | Qualquer | Qualquer | Após $|r|$ alto entre codificações | Correr §7 **antes** de retirar |

**Frase de fecho.** Todo valor codificado é uma estimativa. Trate-o com o **respeito estatístico** que o tamanho amostral e o âmbito de ajuste exigem.

---

## 11. Conclusão

Partimos de dois momentos de entrevista: uma taxa de **100%** em **cinco** linhas, e uma **matriz de correlação** sem regra de decisão. Ambos são sintomas do mesmo descuido — confundir **estimativas** com **verdades**.

O artigo demonstrou: (i) o EMV $\hat{p}_c$ e a sua variância sob baixo suporte; (ii) intervalos Agresti–Coull que mantêm a incerteza visível quando $\hat{p}_c$ atinge 0 ou 1; (iii) suavização Beta–Binomial como média a posteriori com peso transparente dados-versus-prior; (iv) codificações catalogadas pelo **estimando**; (v) uma armadilha de correlação em que a redundância **não** está implícita; (vi) uma heurística orientada ao alvo para decisões de features; (vii) vazamento como âmbito errado de estimação; (viii) quatro figuras reproduzíveis ligando afirmações ao código.

**Takeaways condicionais.** Se $n_c$ é pequeno, suavizar a linguagem e alargar intervalos antes de confiar em taxas extremas. Se dois target encodings correlacionam alto, **medir cada um contra $Y$** antes de retirar um. Se usa target encoding, **definir a amostra** em que $k_c$ e $n_c$ são calculados com o mesmo cuidado que a divisão treino/teste.

**O que vem a seguir.** Aplicar a mesma lente de estimador a codificações **sequenciais** ou de **produção** (deriva de conceito), a priors **hierárquicos** entre categorias relacionadas, e a **monitorização** de $n_c$ no tempo para níveis raros não voltarem silenciosamente a alta variância.

**Limitações.** Todas as afirmações quantitativas ligadas a AUC-PR, F1 e magnitudes de correlação são **condicionais** ao gerador sintético em `src/data.py` e hiperparâmetros em `config.yaml`. O artigo **não** afirma que target suavizado bate sempre one-hot em fraude real, nem que o Experimento C mostrará sempre $|r|>0{,}7$. As afirmações **lógicas** — variância sob baixo suporte, comportamento de intervalos nos limites, vazamento por âmbito errado, e correlação $\not\Rightarrow$ redundância condicional para $Y$ — mantêm-se independentemente do sorteio sintético.

---

## Apêndice A: Mapa de ficheiros do repositório

| Local no artigo | Código / docs |
|-----------------|---------------|
| §§3–4, Fig. 1 | `scripts/experiment_a_perfect_feature.py`, `src/stats_utils.py` |
| §§4–5, Fig. 2 | `scripts/experiment_b_smoothing_effect.py`, `src/encoding.py`, `src/models.py` |
| §§6–7, Fig. 3 | `scripts/experiment_c_correlation_trap.py` |
| §8, Fig. 4 | `scripts/experiment_d_encoding_comparison.py` |
| Gerador | `src/data.py`, `docs/dataset-design.md` |
| Resumo numérico | `docs/experiments-summary.md` |
| Símbolos | `article/notation.md` |
| BibTeX | `article/references.bib` |

Ordem para todas as figuras: `python scripts/run_all.py` (ver `CONTRIBUTING.md` para convenções de branch).

---

## Referências

[1] Agresti, A., & Coull, B. A. (1998). Approximate is better than “exact” for interval estimation of binomial proportions. *The American Statistician*, 52(2), 119–126.

[2] Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101–133.

[3] Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

[4] Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.

[5] Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes. *ACM SIGKDD Explorations*, 3(1), 27–32.

[6] Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining. *ACM TKDD*, 6(4), 1–21.

[7] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

*Chaves BibTeX:* `agresti1998approximate`, `brown2001interval`, `bishop2006pattern`, `murphy2012machine`, `micci2001preprocessing`, `kaufman2012leakage`, `hastie2009elements` — ver `article/references.bib`.
