# Quando uma feature parece boa demais para ser verdade

### Fundamentos estatísticos de engenharia de features categóricas para detecção de fraude — da codificação à inferência

> **Versão para revisão** alinhada ao artigo em inglês: [`features-that-lie.md`](features-that-lie.md).

*Terceira versão — foco em **prática de EDA** (alta proporção no target em categóricas), **preditoras correlacionadas em geral**, e caminhos de figuras compatíveis com GitHub (`../figures/`).*

---

## Resumo

Num case técnico, a **análise exploratória** vai além de listar tipos e missingness. Para variáveis **categóricas** é preciso olhar **com que frequência cada nível aparece** ($n_c$) **e** a **proporção empírica da classe positiva** dentro de cada nível — não tratar uma proporção crua como um facto. Um nível com **taxa observada alta** em **poucas** linhas é, em geral, uma **estimativa de alta variância**, convidando a **viés** e **sobreajuste** se for usada no modelo sem intervalos, suavização ou **âmbito de ajuste** explícito. Em paralelo, quando **duas preditoras** (numéricas, codificadas ou mistas) estão **fortemente correlacionadas**, equipas costumam retirar uma por “colinearidade” sem verificar a **contribuição para o target** em dados de validação. Este artigo unifica os dois pontos: **estatísticas amostrais são estimadores** — a precisão depende de $n_c$ (ou tamanho amostral efectivo) e de se as contagens foram calculadas em treino, teste ou folds (**vazamento**). Revisamos EMV e variância para taxas binomiais, intervalos **Agresti–Coull**, suavização **Beta–Binomial**, codificações pelo **estimando**, e o que inspeccionar quando $|r|$ é grande (**VIF** e coeficientes em modelos lineares; **importância por permutação**, **informação mútua** e **modelos aninhados** de forma mais geral). Quatro experimentos reproduzíveis em dados **sintéticos** ilustram as afirmações. Um **checklist** de produção fecha o ciclo.

**Palavras-chave:** análise exploratória, features categóricas, target encoding, estimação binomial, suavização bayesiana, multicolinearidade, vazamento, detecção de fraude.

---

## 1. Introdução

A maior parte dos fluxos de modelação começa por **análise exploratória**. Abre-se o schema, verificam-se tipos e perfilam-se variáveis. Para colunas **categóricas**, um passo natural é uma tabela de **frequência por nível** e **taxa do target por nível**. Essa tabela é útil — e perigosa se lida de forma ingénua. Um nível com **oitenta ou noventa por cento** de positivos em **vinte** linhas **não** é o mesmo tipo de evidência que o mesmo padrão em **vinte mil** linhas. O primeiro é dominado pela **variância de estimação**; o segundo é muito mais informativo. O erro é **só** “tratar os dados” (codificar e treinar) sem **validar** estatisticamente o que essas proporções significam.

Surge uma segunda lacuna depois de **triagem bivariada**. Matrizes de correlação — Pearson para relações mais lineares, Spearman para monótonas — são calculadas para colunas **numéricas** e, após codificação, para colunas **derivadas**. Quando duas preditoras correlacionam **fortemente**, alguém pode propor retirar uma para reduzir redundância. Isso pode ser correcto ou catastrófico. A **correlação mede associação entre preditoras**; **não** responde, por si só, se ambas as colunas **ajudam a prever** o target, sobretudo em modelos **não lineares** ou com **interacções**.

A postura deste artigo é estatística: **todo o resumo que entra no modelo é um estimador** com variância e uma **amostra correcta** em que deve ser ajustado. Para um nível $c$, a proporção $\hat{p}_c = k_c/n_c$ é um EMV; a variância amostral escala como $1/n_c$. A **suavização** é a média a posteriori Beta–Binomial quando esse é o prior escolhido [3,4]. O **vazamento** é usar informação do rótulo que não existirá em scoring ao construir esses resumos [6].

**O que levar da leitura.** (1) Na EDA, para categóricas com **alta proporção no target**, acompanhar sempre a taxa com **$n_c$**, acrescentar um **intervalo binomial** (p.ex. Agresti–Coull) e considerar **suavização** antes de confiar numa estimativa pontual. (2) Quando **quaisquer** preditoras correlacionam muito, **investigar** com ferramentas orientadas ao alvo (importância por permutação, informação mútua, comparação de modelos em hold-out) e, em modelos **lineares**, **VIF / regularização** — não retirar colunas só por $|r|$. (3) **Categóricas codificadas por target** são um caso especial importante: podem correlacionar forte e ambas continuarem **informativas para $Y$**.

Os **experimentos sintéticos** neste repositório exageram alguns efeitos para os gráficos; em **produção** vêem-se mais frequentemente taxas **altas mas não perfeitas** sob baixo suporte — aplicam-se as **mesmas ferramentas**.

**Roteiro.** §2: enquadramento do estimador. §3: baixo suporte e intervalos. §4: suavização bayesiana e bibliotecas. §5: codificações. §6: **preditoras altamente correlacionadas** (geral + caso target encoding). §7: **o que fazer** após correlações altas. §8: vazamento. §9: experimentos. §10: enquadramento de codificação. §11: conclusão. **Apêndice B:** checklist de produção. **Nota:** figuras com `../figures/` para renderizar no GitHub dentro de `article/`.

---

## 2. Uma estatística ao nível da categoria é um estimador

Se quase nada foi observado no nível $c$, a taxa empírica $\hat{p}_c$ é um **resumo ruidoso**: pode ficar perto de $0{,}9$ ou $1$ ao acaso mesmo quando a taxa positiva de longo prazo é moderada. **Poucas linhas implicam alta variância** em $\hat{p}_c$. Usar essa taxa como feature numérica sem encolhimento ou regularização injecta **entradas de alta variância** e favorece **sobreajuste** em níveis raros.

Formalmente, seja $X$ categórica e $c$ um nível fixo. Entre $n_c$ linhas de treino com $X=c$, seja $k_c$ a contagem com $Y=1$. A **proporção amostral**

$$
\hat{p}_c = \frac{k_c}{n_c}
$$

é o **EMV** de $p_c = P(Y=1 \mid X=c)$ sob binomial [7]. A variância amostral (dado $n_c$) é

$$
\mathrm{Var}(\hat{p}_c) = \frac{p_c(1-p_c)}{n_c}.
$$

**Baixo suporte** significa $n_c$ pequeno, logo variância grande: $\hat{p}_c$ é um **estimador de alta variância**. O valor **observado** pode parecer extremo mesmo quando $p_c$ não o é.

A variância **plug-in** $\hat{p}_c(1-\hat{p}_c)/n_c$ é **zero** quando $\hat{p}_c$ é $0$ ou $1$, sugerindo falsamente certeza. Intervalos **Wilson**, **Agresti–Coull** e **Clopper–Pearson** mantêm-se largos nesse regime [1,2].

**Consequência.** Target encoding que mapeia $c$ para $\hat{p}_c$ não grava um “risco verdadeiro” no nível; passa uma **estimativa** governada por $n_c$.

**Assimptótica.** Para $n_c$ grande, o EMV é aproximadamente normal. Dados de fraude têm muitas vezes **cauda longa** de níveis esparsos — onde atalhos falham. **Nunca tratar $\hat{p}_c$ como igual a $p_c$** quando $n_c$ é pequeno.

---

## 3. O problema de baixo suporte: desconstrução numérica

Na EDA pode aparecer $(k_c, n_c) = (5,5)$ logo $\hat{p}_c=1$, ou $(7,10)$ logo $\hat{p}_c=0{,}7$. **Produção** assemelha-se mais ao segundo padrão do que a 100% literais em cinco linhas; exemplos sintéticos por vezes usam o caso limite porque **dramatiza** a falha plug-in — a **lógica** é a mesma para **qualquer** $\hat{p}_c$ **alta** com $n_c$ **pequeno**.

A proporção ajustada Agresti–Coull usa

$$
\tilde{p} = \frac{k+2}{n+4}, \qquad \tilde{n} = n+4.
$$

Para $(k,n)=(5,5)$, $\tilde{p}=7/9\approx 0.78$. Um intervalo nominal a 95% é $\tilde{p} \pm z_{0.975}\sqrt{\tilde{p}(1-\tilde{p})/\tilde{n}}$, muitas vezes entre **0,5** e **1** após truncagem [1]. Para $(k,n)=(7,10)$, o intervalo continua **largo** face a um nível com milhares de linhas.

Contraste com nível de **$n_c$ alto** (p.ex. $n_c\sim 10^{4}$, $\hat{p}_c$ perto da taxa global): a mesma maquinaria dá faixa **estreita**. A pergunta “qual é $p_c$?” tem **precisão** diferente por nível.

| Perfil | $n_c$ (ilustrativo) | $k_c$ | $\hat{p}_c$ | Intervalo AC 95% (ordem de grandeza) |
|--------|------------------------|---------|----------------|----------------------------------------|
| Esparso, extremo | $\approx 5$ | $\approx 5$ | $1.0$ | Muito largo |
| Esparso, alto | $10$ | $7$ | $0.7$ | Ainda largo |
| Bem suportado | $\approx 10^{4}$ | $\approx 0{,}004\,n_c$ | $\approx 0{,}004$ | Estreito |

**Regra prática.** Reportar sempre **$n_c$** junto de cada taxa por nível nos outputs de EDA. Abaixo de algumas dezenas de linhas (limiar afinado com **domínio** e **desempenho em hold-out**), tratar taxas como **baixo suporte**: mostrar **intervalos**, aplicar **suavização** ou agregar níveis antes de declarar “sinal forte”.

---

## 4. Suavização bayesiana: solução fundamentada

Prior $p_c \sim \mathrm{Beta}(\alpha,\beta)$, verosimilhança $k_c \mid p_c \sim \mathrm{Binomial}(n_c, p_c)$. Posterior

$$
p_c \mid k_c, n_c \sim \mathrm{Beta}(\alpha + k_c,\; \beta + n_c - k_c),
$$

com média

$$
\tilde{p}_c^{\mathrm{Bayes}} = \frac{\alpha + k_c}{\alpha + \beta + n_c}
= w_c\,\hat{p}_c + (1-w_c)\,\mu_0,
\qquad
w_c = \frac{n_c}{n_c + \alpha + \beta}.
$$

$n_c$ pequeno puxa a estimativa para a média a priori $\mu_0$ (frequentemente a taxa global $\bar{p}$); $n_c$ grande recupera o EMV [3,4].

**Bibliotecas.** Parâmetros de suavização em `category_encoders` mapeiam para **força do prior** [5]. `TargetEncoder` do **scikit-learn** (1.2+) usa estatísticas **cross-fitted** — alinhado com **âmbito de ajuste** correcto.

**Pseudocódigo.**

```text
global_mean ← média(y_train)
para cada nível c em X_train:
    (k_c, n_c) ← contagens do nível c no treino
    map[c] ← (alpha + k_c) / (alpha + beta + n_c)
para cada linha i em X_apply:
    encoded[i] ← map[x_i] se x_i em map senão global_mean
```

---

## 5. Panorama das codificações

| Codificação | Fórmula (nível $c$) | Estima | Usa $Y$? | Vazamento | Alta card. |
|-------------|---------------------|--------|----------|-----------|------------|
| One-hot | $\mathbb{1}[X=c]$ | Pertencer a $c$ | Não | Nenhum | Fraco |
| Frequência | $n_c/N$ | $\hat{P}(X=c)$ | Não | Nenhum | Bom |
| Target naïve | $k_c/n_c$ | EMV de $P(Y=1 \mid X=c)$ | Sim | **Alto** se mal feito | Bom |
| Target suavizado | $(k_c+\alpha)/(n_c+\alpha+\beta)$ | Média posterior | Sim | Menor | Bom |

**Quando usar (resumo).** One-hot para $C$ baixa e modelos lineares. Frequência quando a **raridade** de $X$ importa. Target naïve sobretudo como **referência** — produção deve preferir **OOF/CV** ou suavizado. Árvores com categorias **nativas** podem dispensar codificação manual; colunas target podem acrescentar escalar — **validar** em hold-out [7]. **Embeddings** fora de âmbito; os mesmos cuidados de **âmbito** se forem treinados com rótulo.

---

## 6. Preditoras altamente correlacionadas: o que investigar

Correlação **par a par** (Pearson ou Spearman) é achado de **triagem**, não regra de apagar. Aplica-se a **numéricas**, **binárias**, **target encodings** e misturas.

**O que a correlação não diz.** Não diz se retirar uma coluna **melhora ou piora** a predição de $Y$ em **validação**. Duas preditoras podem mover-se juntas e levar informação **não sobreposta** sobre o target (relações **parciais**, **interacções**, efeitos **não lineares** pouco visíveis a Pearson).

**O que ver a seguir (kit geral).**

1. **Ligação ao target:** informação mútua com $Y$, importância por permutação, ou **variação de métrica** em validação ao retirar cada coluna **mantendo a outra**.
2. **Modelos lineares:** **VIF** e estabilidade dos coeficientes; **ridge/elastic net** muitas vezes tratam melhor a redundância do que cortes arbitrários.
3. **Modelos não lineares** (árvores, GBDT): importância e métricas em hold-out pesam mais que $|r|$ entre features; **dependência parcial** ou SHAP com **cuidado** sob correlação.
4. **Correlação parcial** / controlar confundidores quando a história causal sugere causa comum.

**Caso especial: categóricas codificadas por target.** A correlação entre duas colunas target-encoded mede associação linear entre **taxas atribuídas por linha**; **não** implica **redundância condicional** para $Y$. A tabela pequena abaixo é o contraexemplo limpo.

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

Encodings naïves de treino dão $\mathrm{Corr}(z_1,z_2)\approx 0{,}72$, mas $P(Y=1 \mid X_1=A)=1$ e $P(Y=1 \mid X_2=M)=1/4$, e um modelo com **ambas** pode superar cada uma sozinha.

**Conclusão.** Tratar **$|r|$ alto** como estímulo para **diagnósticos orientados ao target** e **comparações de modelo**, não como licença para apagar features.

---

## 7. Depois de ver correlações altas: sequência prática

Quando uma matriz de correlação (ou pairplot) assinalha associação **forte** entre $A$ e $B$, uma ordem de trabalho produtiva é:

1. **Registar** o par e o coeficiente (Pearson vs Spearman).
2. **Pontuar cada preditora contra $Y$** em **validação**: informação mútua, importância por permutação, ou mudança de métrica ao remover cada uma **com a outra presente**.
3. Se **ambas** ajudam materialmente o modelo, **manter ambas** salvo exigência de simplicidade — documentar evidência.
4. Se uma é **inerte** em validação na presença da outra, a candidata a sair é a **inerte** — documentar métricas.
5. **Registar** a decisão com números, não “removemos colineares.”

Para colunas **derivadas do target**, o passo 2 é **obrigatório** antes de qualquer remoção [7].

---

## 8. Vazamento de target

**Vazamento** aqui: a feature usa o **rótulo da própria linha** (ou informação futura) de forma **impossível** em scoring.

**Exemplo mínimo.** Três linhas com nível $c$ e rótulos $(1,0,0)$. Taxa target **naïve** **incluindo** cada linha torna a feature um **proxy de $Y$**. Métricas de treino incham; **teste** diz a verdade. **Correcção:** contagens **out-of-fold**, leave-one-out, ou **só treino** [6].

**Impacto no modelo.** AUC-PR / precisão de treino inflacionados, importância **enganadora**, mau comportamento em **deploy**. **$n_c$** baixo amplifica o efeito de cada rótulo em $\hat{p}_c$.

**Teste rápido.** Treino dispara e validação estagna após features de target → auditar **onde** $k_c$ e $n_c$ foram calculados.

---

## 9. Experimentos

Na **raiz do repositório**:

```text
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
```

Figuras em `figures/` (DPI em `config.yaml`). Ver `docs/dataset-design.md` e `docs/experiments-summary.md`.

**GitHub:** links `../figures/` para abrir `article/features-that-lie.pt-BR.md` na interface web.

| Experimento | Insight | Caminho da figura |
|-------------|---------|-------------------|
| A | Intervalos largos para $n_c$ minúsculo; suavização encolhe para $\bar{p}$ | `../figures/exp_a_perfect_feature.png` |
| B | Target naïve frequentemente maior lacuna treino–teste | `../figures/exp_b_smoothing_effect.png` |
| C | TEs naïve podem correlacionar moderadamente após pooling e ambas seguem informativas; hold-out favorece o modelo conjunto aqui | `../figures/exp_c_correlation_trap.png` |
| D | Pooling com rótulos infla AUC-PR de **treino** | `../figures/exp_d_encoding_comparison.png` |

### Figura 1 — Experimento A

![Figura 1](../figures/exp_a_perfect_feature.png)

**Configuração.** Split estratificado; no código, um **nível sintético espars** (todas as linhas mantidas em treino) com $n_c=5$, $k_c=5$. Intervalos Agresti–Coull; painel com força do prior $m=\alpha+\beta$.

### Figura 2 — Experimento B

![Figura 2](../figures/exp_b_smoothing_effect.png)

XGBoost com numéricos e país como target naïve, suavizado ou one-hot.

### Figura 3 — Experimento C

![Figura 3](../figures/exp_c_correlation_trap.png)

Target naïve para `country` e `merchant_category` só no treino; $r$ de Pearson nas codificações por linha; informação mútua com $Y$; XGBoost com ambas as TEs vs uma removida. Painel (a): pooling em muitos níveis atenua a correlação linear face ao par latente; (b)–(c): métricas em hold-out e ligação ao target antes de retirar colunas.

### Figura 4 — Experimento D

![Figura 4](../figures/exp_d_encoding_comparison.png)

Pipeline com vs sem vazamento por âmbito de codificação.

---

## 10. Enquadramento de codificação

1. Sinalizar **$n_c$** pequeno na EDA; intervalos, suavização ou agregação antes de $\hat{p}_c$ cru.
2. Escolher codificação (§5).
3. Mapas com target: definir **âmbito** (só treino, OOF, CV) antes de afinação.
4. Após correlações, seguir **§7** antes de retirar colunas.
5. Validar em **hold-out**; vigiar lacuna treino–validação.

| Cardinalidade | Suporte | Modelo | Codificação | Âmbito target |
|---------------|---------|--------|-------------|---------------|
| Baixa | Alto/nível | Logística | One-hot ou target suavizado | Treino + CV |
| Alta | Cauda | GBDT nativo | Nativo ± target suavizado | OOF / só treino |
| Alta | Cauda | Rede | Embedding ou frequência | Sem vazamento |
| Qualquer | Qualquer | Qualquer | $|r|$ alto | §7 antes de retirar |

---

## 11. Conclusão

**EDA** em categóricas deve combinar **contagens**, **taxas de target** e **incerteza** (intervalos, suavização). **Alta correlação** entre preditoras deve disparar checagens **orientadas ao target** e **baseadas em modelo**, não remoção automática. Níveis **target-encoded** são um caso em que $|r|$ pode enganar.

**Resumo**

- $\hat{p}_c$ tem variância $\sim 1/n_c$; plug-in falha em $0$ e $1$.
- Intervalos Agresti–Coull quantificam incerteza em níveis esparsos.
- Suavização Beta–Binomial encolhe com análogos em bibliotecas.
- Codificações diferem pelo **estimando**.
- **Correlação $\not\Rightarrow$ redundante para $Y$**; usar **ablation** em validação.
- **Vazamento** parte a generalização; auditar **âmbito**.

Gráficos quantitativos dependem do gerador; o **fluxo de trabalho** não.

**Frase de fecho.** **Perfile categóricas com $n_c$ e intervalos, trate correlação como estímulo para checagens orientadas ao modelo, e só faça deploy de codificações depois de saber de que amostra saiu cada número.**

---

## Apêndice A: Mapa do repositório

| Tema | Local |
|------|--------|
| Fig. 1 | `scripts/experiment_a_perfect_feature.py`, `src/stats_utils.py` |
| Fig. 2 | `scripts/experiment_b_smoothing_effect.py`, `src/encoding.py`, `src/models.py` |
| Fig. 3 | `scripts/experiment_c_correlation_trap.py` |
| Fig. 4 | `scripts/experiment_d_encoding_comparison.py` |
| Dados | `src/data.py`, `docs/dataset-design.md` |
| Resumo métricas | `docs/experiments-summary.md` |
| Notação | `article/notation.md` |
| BibTeX | `article/references.bib` |

`python scripts/run_all.py` na **raiz** regenera todas as figuras.

---

## Apêndice B: Checklist de produção

- [ ] Tabelas de EDA com **$n_c$** e **intervalos** (ou taxas suavizadas) para níveis esparsos.
- [ ] Limiares de baixo suporte documentados; regras de **Other** / agregação.
- [ ] Sem estatísticas target **naïve** ajustadas com rótulos de validação/teste.
- [ ] Target encoding **OOF/CV** ou só treino; política no model card.
- [ ] Pares com $|r|$ alto passam pela **§7** antes de remoções.
- [ ] Métricas de validação com/sem colunas suspeitas registadas.
- [ ] Plano de refresh / deriva para $n_c$ e taxas.

---

## Ver este artigo no GitHub

- **Imagens:** caminhos relativos `../figures/*.png` a partir desta pasta.
- **Equações:** o GitHub renderiza [LaTeX em Markdown](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions) com `$...$` e `$$...$$`. Prefira subscritos simples sem chavetas extra (`$n_c$`, `$\hat{p}_c$`); evite `\{...\}` em math curto quando a prosa for mais legível.
- Para **layout e matemática** de portfólio, gerar **HTML** no repositório portfolio (MathJax).

---

## Referências

[1] Agresti, A., & Coull, B. A. (1998). Approximate is better than “exact” for interval estimation of binomial proportions. *The American Statistician*, 52(2), 119–126.

[2] Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101–133.

[3] Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

[4] Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.

[5] Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes. *ACM SIGKDD Explorations*, 3(1), 27–32.

[6] Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining. *ACM TKDD*, 6(4), 1–21.

[7] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

*Chaves BibTeX:* ver `article/references.bib`.
