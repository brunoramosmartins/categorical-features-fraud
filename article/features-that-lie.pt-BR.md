# Quando uma feature parece boa demais para ser verdade

### Fundamentos estatísticos de engenharia de features categóricas para detecção de fraude — da codificação à inferência

> **Nota:** tradução para revisão alinhada à **terceira versão** (revisão de clareza, narrativa e acessibilidade) do artigo em inglês: [`features-that-lie.md`](features-that-lie.md).

*Terceira versão — revisão de clareza, narrativa e acessibilidade para portfólio.*

---

## Resumo

Durante um caso técnico para uma vaga em detecção de fraude, um praticante analisa uma tabela de taxas de fraude por país. O Uruguai apresenta **100% de taxa de fraude** — em apenas cinco transações. O instinto é confiar nesse número como um sinal poderoso. Mais adiante, uma matriz de correlação entre duas features codificadas retorna $|r| > 0{,}7$; alguém sugere remover uma coluna por "colinearidade." Ambas as decisões parecem razoáveis. Ambas estão erradas.

A causa raiz é a mesma: **confundir uma estatística amostral com um parâmetro populacional.** Toda taxa por categoria inserida em um modelo é um **estimador** — sua precisão depende de quantas linhas a sustentam ($n_c$) e de quais dados foram usados para calculá-la (treino, teste ou fold). Uma taxa de 100% sobre cinco linhas não é um fato sobre o Uruguai; é uma estimativa de alta variância compatível com taxas verdadeiras tão baixas quanto 50%. Uma correlação alta entre duas colunas codificadas não prova que ambas carregam o mesmo sinal sobre o target.

Este artigo torna essa postura estatística operacional. O leitor aprenderá: (1) por que uma categoria com taxa observada extrema e poucas observações é mais provavelmente ruído do que sinal — e como quantificar a incerteza; (2) o que cada codificação comum (one-hot, frequência, target naïve, target suavizado) de fato estima; (3) por que correlação alta entre features codificadas por target não significa que uma é redundante — e o que verificar em vez disso; (4) como decidir quando remover uma feature e quando mantê-la, com um checklist reproduzível; e (5) um framework de decisão para escolhas de codificação baseado em cardinalidade, suporte e tipo de modelo.

Quatro experimentos sobre um dataset sintético de fraude, com todo o código neste repositório, ilustram cada afirmação. Extremos sintéticos são escolhidos por clareza pedagógica; dados de produção mais frequentemente mostram taxas altas-mas-não-perfeitas sob baixo suporte — as mesmas estatísticas se aplicam.

**Palavras-chave:** features categóricas, target encoding, estimação binomial, suavização bayesiana, vazamento (leakage), detecção de fraude, overfitting.

---

## 1. Introdução

Um praticante analisa uma tabela de taxas de fraude por país durante um caso técnico. O Uruguai apresenta 100% de taxa de fraude — cinco transações, todas fraudulentas. O número é impactante, e o instinto é confiar nele: um país onde toda transação observada é fraude deve carregar um sinal forte. Mas a orientação do entrevistador é remover o Uruguai do modelo. Por quê?

Mais adiante, na mesma análise, uma matriz de correlação é calculada entre duas features codificadas — país e categoria do estabelecimento, ambas mapeadas para suas taxas de target. O coeficiente de Pearson retorna acima de 0,7. Alguém sugere remover uma coluna para reduzir colinearidade. Mas qual? E com que justificativa? A análise para na observação — um número é produzido, mas nenhum framework de decisão segue.

Esses dois momentos — uma estatística tomada ao pé da letra e uma métrica calculada sem plano de ação — compartilham a mesma causa raiz. Em ambos os casos, uma **quantidade amostral** é tratada como se fosse uma **verdade populacional**. A taxa de 100% sobre cinco linhas não é um fato sobre o Uruguai; é $\hat{p}_c = k_c / n_c = 5/5$, uma estimativa de máxima verossimilhança com variância inversamente proporcional a $n_c$. O coeficiente de correlação mede associação linear entre duas colunas de taxas codificadas; ele nada diz sobre se remover uma coluna prejudicaria a capacidade do modelo de prever $Y$.

**Toda estatística ao nível da categoria usada em engenharia de features é um estimador** — com um tamanho amostral que governa sua precisão, uma variância que cresce quando $n_c$ diminui, e um escopo de ajuste (treino, teste ou fold) que determina se há vazamento (leakage) da resposta [3,4,6]. A suavização é a média a posteriori Beta–Binomial. O vazamento é calcular $P(Y \mid X = c)$ com informação indisponível no momento da predição. Este artigo torna essa postura de estimador operacional: fórmulas onde esclarecem, experimentos onde convencem, e checklists onde entregam.

*Todos os experimentos utilizam um dataset sintético de fraude projetado para exibir os fenômenos acima. Dados de produção mais frequentemente mostram taxas altas-mas-não-perfeitas sob baixo suporte — as mesmas estatísticas se aplicam.*

---

## 2. Uma estatística ao nível da categoria é um estimador

**Intuição primeiro.** Se quase nada foi observado no nível $c$, a taxa empírica $\hat{p}_c$ é um **mostrador ruidoso**: pode ir a 0,9 ou 1,0 ao acaso mesmo quando a taxa de longo prazo é moderada. **Poucas linhas → alta variância** em $\hat{p}_c$. Engenharia de features que injecta $\hat{p}_c$ no modelo portanto introduz **entradas de alta variância** salvo suavização, regularização ou agregação de informação.

Agora o enquadramento formal. Seja $X$ uma feature categórica e $c$ um nível fixo. Entre as $n_c$ linhas de treino com $X=c$, seja $k_c$ o número com $Y=1$ (fraude). A **proporção amostral**

$$
\hat{p}_c = \frac{k_c}{n_c}
$$

é o **estimador de máxima verossimilhança** (EMV) de $p_c = P(Y=1\mid X=c)$ sob um modelo binomial: condicionado a $X=c$, cada $Y$ é Bernoulli$(p_c)$ [7].

A variância amostral (condicionando em $n_c$) é

$$
\mathrm{Var}(\hat{p}_c) = \frac{p_c(1-p_c)}{n_c}.
$$

**Baixo suporte** significa $n_c$ pequeno, logo variância grande: $\hat{p}_c$ é um **estimador de alta variância sob baixo suporte**. O valor **observado** pode ser extremo mesmo quando $p_c$ é moderado.

Surge patologia separada para estimativas **plug-in** de variância $\hat{p}_c(1-\hat{p}_c)/n_c$: quando $\hat{p}_c\in\{0,1\}$, a expressão é **zero**, sugerindo — falsamente — que não há incerteza. Métodos de intervalo para proporções binomiais (Wilson, Agresti–Coull, Clopper–Pearson) mantêm-se largos nesse regime [1,2].

**Consequência.** Target encoding que mapeia $c$ para $\hat{p}_c$ não produz uma “verdadeira propensão à fraude” gravada na categoria; produz uma **estimativa** cuja confiabilidade é governada por $n_c$. Alimentá-la sem controle a modelos flexíveis aumenta o risco de **overfitting** em níveis raros.

**Assintótica e modelagem.** Para $n_c$ grande, o EMV é aproximadamente normal com a variância acima. Em fraude, uma **cauda longa** de níveis raros muitas vezes conduz a decisões de política — precisamente onde atalhos normais e plug-in falham em conjunto. O takeaway de engenharia: **nunca confundir $\hat{p}_c$ com $p_c$** quando $n_c$ é pequeno.

Mas quão larga é a incerteza para um nível raro concreto? A próxima seção coloca números nisso.

---

## 3. O problema de baixo suporte: uma desconstrução numérica

O exemplo recorrente usa um nível raro **Uruguai** em dados **sintéticos**: cerca de cinco linhas na amostra, todas positivas, logo $k_c=n_c=5$ e $\hat{p}_c=1$. **Pipelines reais** vêem mais frequentemente taxas **altas** (p.ex. 0,75–0,95) com **$n_c$** pequeno; a lógica de intervalos abaixo aplica-se igual — o caso limite $\hat{p}_c=1$ é a ilustração **mais nítida** da falha da variância plug-in, não o único caso relevante.

A proporção ajustada Agresti–Coull usa pseudo-contagens:

$$
\tilde{p} = \frac{k+2}{n+4}, \qquad \tilde{n} = n+4.
$$

Para $(k,n)=(5,5)$, $\tilde{p}=7/9\approx 0.78$. Um intervalo aproximado a 95% usa $\tilde{p} \pm z_{0.975}\sqrt{\tilde{p}(1-\tilde{p})/\tilde{n}}$, produzindo uma faixa larga — frequentemente até cerca de **0,5** e **1** após truncagem [1]. Uma taxa observada perto de **1** é **compatível** com um $p_c$ verdadeiro muito abaixo de 1.

Para ilustração **menos extrema**, tome $(k,n)=(7,10)$: $\hat{p}_c=0{,}7$. O intervalo Agresti–Coull ainda é **largo** face a regimes com milhares de linhas; o ponto é que $\hat{p}_c$ **moderadamente alta** com **dezenas** de linhas ainda carrega incerteza substancial comparado com níveis com milhares de linhas.

Contraste com um país grande como o **Brasil**: se $n_c$ é da ordem de $10^4$ e $\hat{p}_c\approx 0{,}004$, a mesma maquinaria produz uma faixa **estreita** (largura da ordem de $10^{-3}$). O estimador é informativo porque **$n_c$ é informativo**.

Taxas **observadas** extremas com $n_c$ minúsculo são **mais plausivelmente** impulsionadas por **ruído amostral** do que por uma taxa populacional **bem conhecida** — quer o valor observado seja 1,0 ou 0,85.

| Âncora | $n_c$ (ilustrativo) | $k_c$ | $\hat{p}_c$ | Intervalo AC 95% (ordem de grandeza) |
|--------|---------------------|-------|-------------|--------------------------------------|
| Nível raro (sintético) | $\approx 5$ | $\approx 5$ | $1.0$ | Largo (ex.: extremo inferior $\approx 0{,}5$) |
| Nível raro (moderado) | $10$ | $7$ | $0.7$ | Largo (largura substancial) |
| Brasil | $\approx 10^4$ | $\approx 0{,}004\,n_c$ | $\approx 0{,}004$ | Estreito ($\sim 10^{-3}$ largura) |

A linha do Brasil é **ilustrativa**. A **estrutura** é o ponto: “qual é $p_c$?” tem **precisão** diferente por nível.

**Regra prática (não é lei).** Antes de tratar $\hat{p}_c$ como “verdade” da feature, reportar **$n_c$** junto da taxa; para $n_c$ pequeno, preferir **intervalos** ou valores **suavizados**. Muitas equipes tratam $n_c$ abaixo de algumas dezenas (ou mínimo de domínio) como **baixo suporte** para estimação estável de taxa — calibrar o limiar com **desempenho do modelo em hold-out**, não superstição.

Intervalos quantificam o problema; a próxima seção o resolve com uma correção fundamentada.

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

Com $m=\alpha+\beta$ e média a priori $\mu_0=\alpha/(\alpha+\beta)$,

$$
\tilde{p}_c^{\mathrm{Bayes}} = w_c \hat{p}_c + (1-w_c)\mu_0,
\qquad
w_c = \frac{n_c}{n_c + \alpha + \beta}.
$$

Quando $n_c$ é pequeno, $w_c$ é pequeno: a estimativa **encolhe** para $\mu_0$ (p.ex. taxa global de fraude). Quando $n_c$ é grande, $w_c\to 1$: a média a posteriori **segue** o EMV [3,4].

**Bibliotecas (ponte para a prática).** Em `category_encoders`, parâmetros de suavização em encoders tipo target são úteis de ler como **força do prior** relativamente a $n_c$ [5]. No **scikit-learn** 1.2+, `TargetEncoder` faz estatísticas de target **cross-fitted** para reduzir vazamento — alinhado conceptualmente à disciplina de **escopo de ajuste**, mesmo quando a história paramétrica não é Beta–Binomial.

**Pseudocódigo (mapa de níveis suavizado, treino → aplicar).**

```text
global_mean ← média(y_train)
para cada nível c observado em X_train:
    (k_c, n_c) ← contagens de Y=1 e linhas com X=c no treino
    map[c] ← (alpha + k_c) / (alpha + beta + n_c)
para cada linha i em X_apply:
    x ← categoria da linha i
    encoded[i] ← map[x] se x em map senão global_mean
```

**Escolher $(\alpha,\beta)$.** Um default comum fixa $\mu_0=\alpha/(\alpha+\beta)$ na taxa global de **treino** $\bar{p}$, e escolhe $m=\alpha+\beta$ por validação cruzada ou domínio: $m$ maior puxa níveis raros com mais força para $\bar{p}$ [3,4].

**Ligação à produção.** Em scoring, o nível raro $c$ usa $(k_c,n_c)$ de **treino** (ou janela móvel de treino). A suavização estabiliza o valor codificado; **não** elimina a necessidade de monitorar $n_c$ ao longo do tempo.

Com a suavização em mãos, vale a pena recuar e comparar o que diferentes codificações estimam — e quando cada uma é apropriada.

---

## 5. O panorama das codificações: o que cada uma estima?

| Codificação | Fórmula (nível $c$) | Estima | Usa $Y$? | Risco de vazamento | Alta cardinalidade |
|-------------|---------------------|--------|----------|-------------------|-------------------|
| One-hot | $\mathbb{1}[X=c]$ | Pertencer a $c$ | Não | Nenhum | Fraco (esparsa larga) |
| Frequência | $n_c/N$ | $\hat{P}(X=c)$ | Não | Nenhum | Bom (uma coluna) |
| Target (naïve) | $k_c/n_c$ | $P(Y{=}1\mid X{=}c)$ EMV | Sim | **Alto** se mal ajustado | Bom |
| Target suavizado | $(k_c{+}\alpha)/(n_c{+}\alpha{+}\beta)$ | Mesmo estimando, média posterior | Sim | Reduzido vs extremos; ainda mal escopo | Bom |

**Quando usar (compacto).**

- **One-hot:** cardinalidade baixa, modelos lineares, efeitos interpretáveis por nível; evitar em $C$ muito alto sem regularização.
- **Frequência:** alta cardinalidade, sinal quando a **raridade** de $X$ importa; sem vazamento de rótulo pelo mapa.
- **Target naïve:** raramente adequado para treino final sem **OOF / CV**; útil como contraste pedagógico.
- **Target suavizado:** alta cardinalidade com sinal do rótulo; afinar suavização; definir sempre **escopo de ajuste** (só treino ou OOF).

Boosters com categorias **nativas** procuram divisões $X\in S$; **não** é obrigatória codificação numérica manual. Codificações tipo target podem acrescentar um **escalar** de $P(Y\mid X{=}c)$; validar em hold-out [7].

**Embeddings** estão fora de escopo; se treinados com rótulos, aplicam-se as mesmas questões de **estimador + escopo**.

Saber o que cada codificação estima é necessário — mas não suficiente. Um próximo passo comum é calcular correlações entre colunas codificadas e usá-las para seleção de features. Esse passo esconde uma armadilha.

---

## 6. Correlação não é redundância

Na prática, equipes calculam **matriz de correlação** sobre colunas codificadas, veem $|r|$ grande e param — sem perguntar o que acontece ao **modelo** se uma coluna sai.

**Fato.** A correlação de Pearson entre duas colunas **codificadas por target** mede **associação linear entre linhas** entre as taxas por nível atribuídas. **Não** implica **redundância condicional** para $Y$: $P(Y\mid X_1)$ e $P(Y\mid X_2)$ podem diferir fortemente para pares de níveis mesmo quando as colunas correlacionam.

**Brinquedo (oito linhas).** Níveis $X_1\in\{A,B,C\}$, $X_2\in\{M,N,P\}$, $Y$ binário:

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

Targets naïves de treino dão $\mathrm{Corr}(z_1,z_2)\approx 0{,}72$, mas $P(Y{=}1\mid X_1{=}A)=1$ enquanto $P(Y{=}1\mid X_2{=}M)=1/4$, e um modelo logístico com **ambas** as codificações pode superar qualquer uma sozinha nesta tabela.

**Conclusão principal (modelos).** **Não retire uma feature de um modelo preditivo só porque correlaciona com outra** até comparar **modelos com e sem** essa feature (ou usar pontuações orientadas ao alvo, p.ex. importância por permutação). Correlação marginal **não** substitui **contribuição em hold-out para $Y$**.

**Experimentos em pipeline.** No dataset sintético (§9), a correlação TE por linha pode ficar **abaixo** de $0{,}7$ quando muitos níveis agregam linhas. O **princípio** mantém-se: usar **métricas de modelo** e o checklist da §7, não $|r|$ sozinho.

Se correlação sozinha não basta para justificar remover uma coluna, o que um praticante deve fazer depois de calcular correlações? A próxima seção oferece um checklist concreto.

---

## 7. Depois das correlações: checklist de decisão

Usar **depois** de calcular correlações par a par entre colunas codificadas (em especial derivadas do target).

**Não fazer**

- Retirar uma coluna só porque $|r|>0{,}9$ **sem** pontuar a ligação com $Y$.

**Fazer**

1. Sinalizar pares com $|r|$ alto.
2. Para cada par, pontuar **cada** coluna contra $Y$ (informação mútua, importância por permutação, ou comparação aninhada de modelos).
3. Se **ambas** movem métricas de validação, **manter ambas** salvo simplicidade — **documentar** o trade-off.
4. Se uma é inerte em hold-out, considerar retirar **essa** — **documentar** a métrica.
5. **Escrever a decisão** com números (métricas, $n_c$, política de folds) — não “removemos colineares.”

**Triagens mais seguras** perguntam se a feature **altera predições de $Y$**, não só se acompanha outra [7].

Resta um último modo de falha: mesmo uma codificação bem escolhida e bem avaliada pode quebrar a generalização se foi calculada sobre os dados errados.

---

## 8. Vazamento de target: quando o estimador vê a resposta

**Vazamento (sentido restrito):** o valor de uma feature para uma linha **usa o rótulo dessa linha** (ou rótulos futuros) de forma que **não** pode ocorrer em scoring.

**Exemplo mínimo (três linhas, um nível).** Níveis $\{c,c,c\}$, rótulos $(1,0,0)$. Taxa alvo **naïve** para $c$ calculada **incluindo** a linha dá a cada linha uma codificação que **depende do seu próprio** $Y$. A **função de perda de treino** pode parecer excelente porque o modelo vê um **proxy de $Y$** dentro da feature; o desempenho em **teste** é o juiz honesto — e codificação **correta** usa contagens **excluindo** a linha (OOF/LOO) ou escopos **só treino** [6].

**Impacto no modelo.** O vazamento **infla métricas de treino** (AUC-PR, precisão) e produz **importância de features enganadora**; **não** cria informação disponível em deploy, logo a história de **generalização** parte-se.

**Mecanismo (target naïve).** Agregar rótulos de treino+teste (ou fold) em $k_c$ permite que a codificação **codifique a chave de respostas** para linhas no pool. A correcção é estatísticas **out-of-fold** ou **estritamente só treino** para mapas de codificação [6].

**Baixo suporte amplia a gravidade.** $n_c$ minúsculo implica que um rótulo move $\hat{p}_c$ fortemente — auto-influência grande.

**Teste de detecção rápido.** Se o AUC-PR de treino dispara e o de validação mal se move após adicionar target encoding, **primeiro** auditar **onde** $k_c$ e $n_c$ foram calculados.

A teoria está agora completa. Os experimentos a seguir colocam cada afirmação em teste reproduzível.

---

## 9. Experimentos

**Reprodutibilidade.** A partir da **raiz do repositório**, com Python 3.10+:

```text
python -m venv .venv && source .venv/bin/activate   # ou Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_all.py
```

As figuras são gravadas em `figures/` ao DPI em `config.yaml` (default 300). O gerador está em `docs/dataset-design.md`; ressalvas em `docs/experiments-summary.md`.

Cada experimento abaixo testa uma afirmação específica das seções anteriores.

| Experimento | Insight principal | Figura |
|-------------|-------------------|----------------------------------|
| A | Intervalos largos para $n_c$ minúsculo; suavização puxa níveis raros para $\bar{p}$ | `figures/exp_a_perfect_feature.png` |
| B | Target naïve frequentemente maior lacuna treino–teste que suavizado; ranking varia com seed | `figures/exp_b_smoothing_effect.png` |
| C | $|r|$ alto não justifica retirar coluna sem checagens ao modelo / orientadas ao alvo | `figures/exp_c_correlation_trap.png` |
| D | Agregação com vazamento infla AUC-PR de **treino** vs escopo OOF/só treino correto | `figures/exp_d_encoding_comparison.png` |

### Experimento A — A ilusão da “feature perfeita”

**Configuração.** Split estratificado; **todas** as linhas do Uruguai em **treino** para $n_c{=}5$, $k_c{=}5$. $\hat{p}_c$ naïve e intervalos Agresti–Coull a 95% no treino. Segundo painel: média posterior para o Uruguai vs força do prior $m=\alpha+\beta$.

**Figura 1** (`figures/exp_a_perfect_feature.png`). **Esquerda:** $\hat{p}_c$ vs $n_c$ (log) com barras; Uruguai anotado. **Direita:** média suavizada vs $m$, linha em $\bar{p}$.

**Observação.** Intervalo do Uruguai mantém-se largo com $\hat{p}_c{=}1$; $m$ maior encolhe para $\bar{p}$.

**Ligação teórica.** §§2–4.

### Experimento B — Suavização e generalização

**Configuração.** XGBoost com numéricos mais **país** como target naïve, target suavizado (`config.yaml`), ou one-hot. AUC-PR treino/teste e F1 a 0,5.

**Figura 2** (`figures/exp_b_smoothing_effect.png`). AUC-PR de teste por codificação; barras treino vs teste.

**Observação.** Target naïve mostra frequentemente **maior** lacuna treino–teste; ranking varia.

**Ligação teórica.** §§4–5.

### Experimento C — Correlação versus redundância

**Configuração.** TE naïve para `country` e `merchant_category` **só no treino**; Pearson $r$; IM com $Y$; XGBoost com ambas vs uma coluna retirada.

**Figura 3** (`figures/exp_c_correlation_trap.png`). Dispersão; AUC-PR de teste para {ambas, só país, só merchant}; barras IM.

**Observação.** Não retirar só por $|r|$; gráficos ligam afirmações ao **modelo**.

**Ligação teórica.** §§6–7.

### Experimento D — Vazamento via escopo

**Configuração.** **Com vazamento:** TE com rótulos **concatenados** treino+teste. **Correcto:** OOF no treino; teste com estatísticas **só treino**. Ver `scripts/experiment_d_encoding_comparison.py`.

**Figura 4** (`figures/exp_d_encoding_comparison.png`). AUC-PR treino vs teste: com vs sem vazamento.

**Observação.** AUC-PR de treino pode ser **optimista** com vazamento.

**Ligação teórica.** §8; [6].

---

## 10. Um enquadramento para decisões de codificação

**Eixos.** **Cardinalidade**, **suporte** ($n_c$), **família de modelo** (linear/NN vs categorias nativas).

**Fluxo de decisão (alto nível).**

1. Listar níveis com **$n_c$** pequeno; encaminhar para intervalos, suavização ou agregação — não só estimativas pontuais cruas.
2. Escolher codificação: ver tabela da §5 + “quando usar.”
3. Se usar mapa **baseado em target**: definir **escopo de ajuste** (só treino, OOF ou CV) **antes** de afinar.
4. Depois de matrizes de correlação sobre codificações: correr **checklist da §7** antes de retirar colunas.
5. Validar em **hold-out**; vigiar lacuna **treino vs validação**.

**Heurística (bullets).**

- Alta cardinalidade + cauda pesada de $n_c$ pequeno: preferir **target suavizado** ou **frequência** + modelos regularizados.
- Linear / NN: **one-hot** ou **tipo target** com **cross-fitting**.
- Árvores boost com cats nativas: codificação opcional; validar colunas target acrescentadas.
- Após $|r|$ alto entre codificações: **§7** antes de retirar.

**Tabela bússola** (não é lei — validar sempre em hold-out):

| Cardinalidade | Suporte típico | Modelo (exemplos) | Codificação de primeira linha | Âmbito para mapas com target |
|---------------|----------------|-------------------|------------------------------|------------------------------|
| Baixa | Alto por nível | Regressão logística | One-hot ou target suavizado | Só treino; CV para target |
| Alta | Cauda pesada | XGBoost (cat nativo) | Nativo + opcional target suavizado | OOF / só treino |
| Alta | Cauda pesada | Rede neural | Embedding ou frequência + numérico | Evitar vazamento de rótulo no treino |
| Qualquer | Qualquer | Qualquer | $|r|$ alto entre codificações | §7 **antes** de retirar |

---

## 11. Conclusão

A taxa do Uruguai que parecia um sinal era variância. A correlação que parecia redundância era variação compartilhada. Ambas as respostas vêm do mesmo lugar: tratar uma estimativa como verdade — e ambas se resolvem pela mesma postura: **todo valor codificado é um estimador, governado pelo tamanho amostral e pelo escopo de ajuste.**

Este artigo traçou essa postura dos fundamentos à prática:

- O EMV $\hat{p}_c$ tem **variância** inversamente proporcional a $n_c$; estimativas plug-in falham nos limites onde $\hat{p}_c \in \{0, 1\}$.
- **Intervalos Agresti–Coull** expõem incerteza mesmo para $\hat{p}_c$ moderadamente alta, não apenas para 100%.
- **Suavização Beta–Binomial** encolhe estimativas de níveis raros em direção à taxa global — uma correção transparente com mapeamento direto para bibliotecas.
- Codificações diferem pelo **que estimam**; escolher uma é escolher um estimando.
- **Correlação não implica redundância** para prever $Y$; apenas comparações de modelo e o checklist da §7 respondem a essa pergunta.
- **Vazamento (leakage)** é codificar com o escopo errado — e $n_c$ baixo amplifica o dano.

**Quando agir:**

- $n_c$ pequeno: acompanhar taxas com **intervalos** ou **suavização**; monitorar overfitting.
- $|r|$ alto entre target encodings: **pontuar cada coluna contra $Y$** e comparar modelos antes de remover.
- Qualquer target encoding: definir a **amostra** para $(k_c, n_c)$ com a mesma disciplina do split treino/teste.

Resultados quantitativos (AUC-PR, F1, $r$) dependem do gerador sintético em `src/data.py` e `config.yaml`; as afirmações lógicas — variância, intervalos, vazamento, correlação versus redundância no modelo — não.

**Trate cada coluna codificada como uma estimativa ligada a um tamanho amostral e a um escopo de ajuste — depois valide o modelo que efetivamente vai para produção.**

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

Todas as figuras: `python scripts/run_all.py` a partir da raiz do repositório.

---

## Apêndice B: Checklist orientado a produção

Usar juntamente com §7 e §10 antes de fundir alterações de codificação.

- [ ] Para cada categórica de alto impacto, reportar **$n_c$** por nível (ou monitorar em dashboards).
- [ ] Definir limiares de **baixo suporte** por feature; encaminhar níveis raros para **Other**, suavização ou pooling hierárquico.
- [ ] Nunca fazer deploy de estatísticas target **naïve** ajustadas com rótulos de **validação/teste**.
- [ ] Preferir target encoding **OOF/CV** ou mapas **só treino**; registar a política em model cards.
- [ ] Após análise de correlação sobre codificações, completar a **§7** antes de retirar colunas.
- [ ] Comparar métricas de **validação** com e sem colunas suspeitas; vigiar lacuna **treino–validação**.
- [ ] Revalidar codificações sob **refresh** ou **deriva** (contagens e taxas mudam no tempo).

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
