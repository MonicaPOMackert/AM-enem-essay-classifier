"""
=============================================================
SELEÇÃO DE ATRIBUTOS (Feature Selection) — Yuri, versão Python
=============================================================
Equivalente Python da aba "Select Attributes" do Weka.

DIFERENÇA IMPORTANTE em relação ao PCA que já fizemos no Tier 3:
 - PCA cria colunas NOVAS (combinações lineares) — perde
   interpretabilidade.
 - Feature Selection ESCOLHE colunas ORIGINAIS e descarta as
   piores — mantém 100% de interpretabilidade. É exatamente o
   que reforça a narrativa "Tier 1/2 interpretável vs BERT caixa-preta".

Esse script reproduz os dois componentes que o Weka usa:

 1) ATTRIBUTE EVALUATOR (a "nota" de cada atributo):
    - Equivalente ao GainRatioAttributeEval do Weka -> usamos
      mutual_info_classif (Ganho de Informação / Informação Mútua)
    - Equivalente ao CorrelationAttributeEval -> usamos f_classif
      (ANOVA F-value, mede relação estatística atributo x classe)

 2) SEARCH METHOD (como navegar pelas combinações):
    - Equivalente ao Ranker do Weka -> SelectKBest (pega os K
      melhores atributos pelo ranking, sem testar combinações)
    - Equivalente ao BestFirst do Weka -> RFE (Recursive Feature
      Elimination) com Random Forest, que testa subconjuntos
      removendo o atributo mais fraco a cada rodada

O script roda os dois Tiers (1 e 2) com as duas abordagens,
e testa se reduzir atributos melhora ou prejudica a acurácia.

ANTES DE RODAR:
 pip install pandas numpy scikit-learn
=============================================================
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, f_classif, RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer

# ── AJUSTE AQUI: troque pelos arquivos que já temos prontos ───
ARQUIVOS = {
    'Tier 1 reforçado': 'tier1_reforcado_500.csv',
    'Tier 2 reforçado': 'tier2_reforcado.csv',   # gere com o script 5 se ainda não tiver
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
kappa_scorer = make_scorer(cohen_kappa_score)
rf_avaliador = RandomForestClassifier(n_estimators=100, random_state=42)

def avaliar_modelo(X, y, nome):
    acc = cross_val_score(rf_avaliador, X, y, cv=cv, scoring='accuracy').mean()
    kappa = cross_val_score(rf_avaliador, X, y, cv=cv, scoring=kappa_scorer).mean()
    print(f"{nome:55s} | Acurácia: {acc*100:.2f}%  | Kappa: {kappa:.4f}")
    return {'experimento': nome, 'acuracia': round(acc*100, 2), 'kappa': round(kappa, 4)}

resultados_globais = []

for nome_tier, caminho in ARQUIVOS.items():
    print(f"\n{'='*70}")
    print(f"PROCESSANDO: {nome_tier}")
    print(f"{'='*70}")

    try:
        df = pd.read_csv(caminho)
    except FileNotFoundError:
        print(f"⚠️  Arquivo '{caminho}' não encontrado. Pulando {nome_tier}.")
        continue

    X = df.drop(columns=['score_class'])
    y = df['score_class']
    nomes_colunas = X.columns.tolist()
    X_vals = X.values

    print(f"Total de atributos originais: {X_vals.shape[1]}")

    # ── BASELINE: modelo com TODOS os atributos ────────────────
    resultados_globais.append(
        avaliar_modelo(X_vals, y, f"{nome_tier} — TODOS os atributos (baseline)")
    )

    # ═══════════════════════════════════════════════════════════
    # MÉTODO 1 — Ranker (equivalente Weka: Ranker + GainRatio)
    # Usa Informação Mútua (Ganho de Informação) + seleciona top K
    # ═══════════════════════════════════════════════════════════
    for K in [50, 100, 200]:
        if K >= X_vals.shape[1]:
            continue
        print(f"\n--- Ranker (Informação Mútua) — Top {K} atributos ---")
        selector = SelectKBest(score_func=mutual_info_classif, k=K)
        X_selecionado = selector.fit_transform(X_vals, y)

        # Mostra os 10 melhores atributos (nomes reais — interpretável!)
        scores = selector.scores_
        ranking = sorted(zip(nomes_colunas, scores), key=lambda x: -x[1])
        print(f"Top 10 atributos mais informativos: {[r[0] for r in ranking[:10]]}")

        resultados_globais.append(
            avaliar_modelo(X_selecionado, y, f"{nome_tier} — Ranker top {K} (Info. Mútua)")
        )

    # ═══════════════════════════════════════════════════════════
    # MÉTODO 2 — Ranker com ANOVA F-value (equivalente Weka: CorrelationAttributeEval)
    # ═══════════════════════════════════════════════════════════
    K_anova = min(100, X_vals.shape[1] - 1)
    print(f"\n--- Ranker (ANOVA F-value / Correlação) — Top {K_anova} atributos ---")
    selector_f = SelectKBest(score_func=f_classif, k=K_anova)
    X_selecionado_f = selector_f.fit_transform(X_vals, y)
    scores_f = selector_f.scores_
    ranking_f = sorted(zip(nomes_colunas, scores_f), key=lambda x: -x[1])
    print(f"Top 10 atributos por correlação com a classe: {[r[0] for r in ranking_f[:10]]}")

    resultados_globais.append(
        avaliar_modelo(X_selecionado_f, y, f"{nome_tier} — Ranker top {K_anova} (ANOVA/Correlação)")
    )

    # ═══════════════════════════════════════════════════════════
    # MÉTODO 3 — BestFirst-like (RFE: Recursive Feature Elimination)
    # Mais pesado computacionalmente — testa subconjuntos removendo
    # o atributo mais fraco a cada rodada, igual ao espírito do BestFirst
    # ═══════════════════════════════════════════════════════════
    N_RFE = min(50, X_vals.shape[1] - 1)
    print(f"\n--- BestFirst-like (RFE) — reduzindo para {N_RFE} atributos ---")
    print("(Isso pode demorar alguns minutos, dependendo do tamanho do Tier...)")

    rf_rfe = RandomForestClassifier(n_estimators=50, random_state=42)  # menor pra RFE não ficar muito lento
    rfe = RFE(estimator=rf_rfe, n_features_to_select=N_RFE, step=0.1)
    X_rfe = rfe.fit_transform(X_vals, y)

    colunas_selecionadas_rfe = [nomes_colunas[i] for i in range(len(nomes_colunas)) if rfe.support_[i]]
    print(f"Atributos selecionados pelo RFE (primeiros 10): {colunas_selecionadas_rfe[:10]}")

    resultados_globais.append(
        avaliar_modelo(X_rfe, y, f"{nome_tier} — BestFirst/RFE ({N_RFE} atributos)")
    )

# ── Tabela final ────────────────────────────────────────────────
print(f"\n{'='*70}")
print("TABELA COMPARATIVA FINAL — SELEÇÃO DE ATRIBUTOS")
print(f"{'='*70}")
resumo = pd.DataFrame(resultados_globais)
print(resumo.to_string(index=False))
resumo.to_csv('resultado_selecao_atributos.csv', index=False)
print("\n✅ Salvo: resultado_selecao_atributos.csv")

print("""
=============================================================
COMO INTERPRETAR PRA APRESENTAÇÃO:
=============================================================
- Se a acurácia com MENOS atributos for parecida (ou até melhor)
  que com TODOS os atributos, isso é uma vitória clara: significa
  que muitas colunas eram redundantes ou até prejudicavam o modelo
  (ruído), e dá pra simplificar o modelo sem perder desempenho.

- Equivalência com o Weka:
  * "Ranker" no Python = SelectKBest (escolhe os K melhores
    individualmente, sem considerar interação entre atributos)
  * "BestFirst" no Python = RFE (testa subconjuntos, removendo
    o mais fraco a cada rodada — mais lento, mas considera
    interações entre atributos)

- Pro Tier 1 e 2: como os atributos têm NOMES REAIS (palavras do
  TF-IDF, classes gramaticais), o ranking dos "top atributos" é
  uma ótima informação pra apresentação — mostra exatamente quais
  palavras/POS-tags mais influenciam a nota da redação.
=============================================================
""")
