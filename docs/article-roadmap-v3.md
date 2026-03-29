# Article roadmap (v3) — alinhamento com EDA e correlação geral

Documento de planeamento **versionado no repositório** (substitui/complementa o foco narrativo do `roadmap-categorical-features-article-v2.md` local, se o mantiveres fora do git).

## Perguntas centrais do artigo

1. **EDA — categórica com alta proporção no target**  
   Não é só “pode ser sinal”: que **cuidados** tomar (suporte $n$, âmbito da amostra, leakage), que **estatísticas** ajudam a **validar** (intervalos para proporções, Agresti–Coull / Wilson, suavização como encolhimento, comparação com taxa global), e como isso liga a **encoding** e ao modelo.

2. **Preditoras altamente correlacionadas (qualquer tipo)**  
   O que **investigar** quando $|r|$ (ou Spearman) é alto entre **numéricas**, **codificadas**, ou misturas: contribuição para $Y$ (importância por permutação, MI, nested models), **VIF / regularização** em modelos lineares, **comparação em hold-out**, caso especial de **target encoding** onde correlação $\neq$ redundância condicional.

## O que o repositório entrega

- Texto canónico: `article/features-that-lie.md` (EN) e `article/features-that-lie.pt-BR.md` (PT).  
- Figuras: `figures/*.png`; no Markdown sob `article/`, links **`../figures/...`** para renderizar no GitHub.  
- Matemática: subscritos explícitos `_{c}` em modo inline onde o parser do GitHub confunde `_` com ênfase.  
- HTML “bonito” continua a ser responsabilidade do **repositório portfolio** (MathJax completo, CSS).

## Fases (roadmap clássico)

| Fase | Conteúdo |
|------|-----------|
| 4 | Artigo completo em Markdown + notação + refs |
| 5 | Revisão, alt text, `verify_publication_ready`, portfolio HTML |
| 6 | Publicação / LinkedIn |

## Changelog de foco (v2 → v3)

- Menos anedota “país + 100%”; mais **fluxo de EDA** e **validação estatística**.  
- Correlação como tema **geral**, com categóricas codificadas como caso especial.  
- Menos rótulos estilo “documento legal” (**Contexto/Problema** ou **Não fazer/Fazer**); mais prosa + listas numeradas quando útil.
