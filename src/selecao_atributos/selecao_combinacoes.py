"""
=============================================================
SELEÇÃO DE ATRIBUTOS SOBRE COMBINAÇÕES DE TIERS — Yuri (7b)
=============================================================
Extensão do script 7 (seleção de atributos) aplicada às 3
combinações de Tiers testadas no script 12:

 1) Tier 1 Reforçado + Tier 2 Reforçado            (n=1542)
 2) Tier 1 + Tier 2 + Tema (combo interpretável)    (n=1543)
 3) Tier 1 Reforçado + Tier 3 BERT (PCA)            (n=697)

MOTIVAÇÃO: no script 12, nenhuma combinação superou claramente
o melhor Tier isolado (Tier 3: 58,73% / Kappa 0,333). Uma
hipótese é que a concatenação simples introduz ATRIBUTOS
REDUNDANTES OU RUIDOSOS que prejudicam o modelo. Este script
testa se aplicar seleção de atributos (igual ao script 7
original) sobre essas combinações melhora o resultado.

Mesma lógica do script 7:
 - Ranker (Informação Mútua)  -> SelectKBest(mutual_info_classif)
 - Ranker (ANOVA/Correlação)  -> SelectKBest(f_classif)
 - BestFirst-like (RFE)       -> RFE com Random Forest

PRÉ-REQUISITO: ter na mesma pasta:
 - tier1_reforcado_500.csv
 - tier2_reforcado.csv
 - feature_tema.csv
 - redacoes_tier3_bert.csv (ou redacoes_tier3_bert_processado.csv)
 (os mesmos arquivos usados pelo script 12)

ANTES DE RODAR:
 pip install pandas numpy scikit-learn
=============================================================
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
kappa_scorer = make_scorer(cohen_kappa_score)
rf_avaliador = RandomForestClassifier(n_estimators=100, random_state=42)


def avaliar_modelo(X, y, nome):
    acc = cross_val_score(rf_avaliador, X, y, cv=cv, scoring='accuracy').mean()
    kappa = cross_val_score(rf_avaliador, X, y, cv=cv, scoring=kappa_scorer).mean()
    print(f"{nome:65s} | Acurácia: {acc*100:.2f}%  | Kappa: {kappa:.4f}")
    return {'experimento': nome, 'acuracia': round(acc * 100, 2), 'kappa': round(kappa, 4)}


def rodar_selecao(X_vals, y, nomes_colunas, nome_combo, resultados_globais):
    """Roda os 3 métodos de seleção (mesma lógica do script 7) sobre uma combinação."""
    print(f"\n{'='*70}")
    print(f"PROCESSANDO: {nome_combo}")
    print(f"{'='*70}")
    print(f"Total de atributos originais: {X_vals.shape[1]}")

    # BASELINE: já temos do script 12 (todos os atributos), mas recalculamos
    # aqui pra garantir comparação direta com o mesmo cv/modelo deste script
    resultados_globais.append(
        avaliar_modelo(X_vals, y, f"{nome_combo} — TODOS os atributos (baseline)")
    )

    # ── MÉTODO 1 — Ranker (Informação Mútua) ──────────────────
    for K in [50, 100, 200]:
        if K >= X_vals.shape[1]:
            continue
        print(f"\n--- Ranker (Informação Mútua) — Top {K} atributos ---")
        selector = SelectKBest(score_func=mutual_info_classif, k=K)
        X_selecionado = selector.fit_transform(X_vals, y)
        scores = selector.scores_
        ranking = sorted(zip(nomes_colunas, scores), key=lambda x: -x[1])
        print(f"Top 10 atributos mais informativos: {[r[0] for r in ranking[:10]]}")
        resultados_globais.append(
            avaliar_modelo(X_selecionado, y, f"{nome_combo} — Ranker top {K} (Info. Mútua)")
        )

    # ── MÉTODO 2 — Ranker (ANOVA F-value) ──────────────────────
    K_anova = min(100, X_vals.shape[1] - 1)
    print(f"\n--- Ranker (ANOVA F-value / Correlação) — Top {K_anova} atributos ---")
    selector_f = SelectKBest(score_func=f_classif, k=K_anova)
    X_selecionado_f = selector_f.fit_transform(X_vals, y)
    scores_f = selector_f.scores_
    ranking_f = sorted(zip(nomes_colunas, scores_f), key=lambda x: -x[1])
    print(f"Top 10 atributos por correlação com a classe: {[r[0] for r in ranking_f[:10]]}")
    resultados_globais.append(
        avaliar_modelo(X_selecionado_f, y, f"{nome_combo} — Ranker top {K_anova} (ANOVA/Correlação)")
    )

    # ── MÉTODO 3 — BestFirst-like (RFE) ─────────────────────────
    N_RFE = min(50, X_vals.shape[1] - 1)
    print(f"\n--- BestFirst-like (RFE) — reduzindo para {N_RFE} atributos ---")
    print("(Isso pode demorar alguns minutos — RFE é o método mais pesado.)")
    rf_rfe = RandomForestClassifier(n_estimators=50, random_state=42)
    rfe = RFE(estimator=rf_rfe, n_features_to_select=N_RFE, step=0.1)
    X_rfe = rfe.fit_transform(X_vals, y)
    colunas_selecionadas_rfe = [nomes_colunas[i] for i in range(len(nomes_colunas)) if rfe.support_[i]]
    print(f"Atributos selecionados pelo RFE (primeiros 10): {colunas_selecionadas_rfe[:10]}")
    resultados_globais.append(
        avaliar_modelo(X_rfe, y, f"{nome_combo} — BestFirst/RFE ({N_RFE} atributos)")
    )


resultados_globais = []

print("Carregando arquivos...")
tier1 = pd.read_csv('tier1_reforcado_500.csv')
tier2 = pd.read_csv('tier2_reforcado.csv')
tema = pd.read_csv('feature_tema.csv')
print(f"Tier 1: {tier1.shape}  |  Tier 2: {tier2.shape}  |  Tema: {tema.shape}")

y = tier1['score_class']  # mesma ordem de linhas em todos os arquivos
X_t1 = tier1.drop(columns=['score_class'])
X_t2 = tier2.drop(columns=['score_class'])

# ── Combinação 1: Tier 1 + Tier 2 ───────────────────────────────
X_combo12 = pd.concat([X_t1, X_t2], axis=1)
rodar_selecao(X_combo12.values, y, X_combo12.columns.tolist(),
              "Tier 1 Reforçado + Tier 2 Reforçado", resultados_globais)

# ── Combinação 2: Tier 1 + Tier 2 + Tema ────────────────────────
X_combo12_tema = pd.concat([X_t1, X_t2, tema[['feature_aderencia_tema']]], axis=1)
rodar_selecao(X_combo12_tema.values, y, X_combo12_tema.columns.tolist(),
              "Tier 1 + Tier 2 + Tema", resultados_globais)

# ── Combinação 3: Tier 1 + Tier 3 BERT (PCA) ────────────────────
# OBS: as 184 colunas do PCA não têm nomes interpretáveis (são
# combinações lineares) — usamos rótulos genéricos "bert_pca_N"
# só para o RFE/ranking funcionar; lembrar disso na hora de
# escrever o slide (não dá pra dizer "a componente X é mais
# importante" com significado real, só "componentes do BERT-PCA
# pesam mais que atributos do Tier 1", ou vice-versa).
try:
    tier3 = pd.read_csv('redacoes_tier3_bert.csv')
    print(f"\nTier 3 carregado: {tier3.shape}")
    X_t3 = tier3.drop(columns=['score_class'])

    scaler = StandardScaler()
    X_t3_scaled = scaler.fit_transform(X_t3.values)
    pca = PCA(n_components=0.95, random_state=42)
    X_t3_pca = pca.fit_transform(X_t3_scaled)
    print(f"Tier 3 após PCA: {X_t3_pca.shape}")

    colunas_t3_pca = [f"bert_pca_{i}" for i in range(X_t3_pca.shape[1])]
    X_t3_pca_df = pd.DataFrame(X_t3_pca, columns=colunas_t3_pca)

    X_combo_t1_t3 = pd.concat([X_t1.reset_index(drop=True), X_t3_pca_df], axis=1)
    rodar_selecao(X_combo_t1_t3.values, y, X_combo_t1_t3.columns.tolist(),
                  "Tier 1 Reforçado + Tier 3 BERT (PCA)", resultados_globais)
except FileNotFoundError:
    print("\n⚠️  redacoes_tier3_bert.csv não encontrado — pulando combinação com Tier 3")

# ── Tabela final ────────────────────────────────────────────────
print(f"\n{'='*70}")
print("TABELA COMPARATIVA FINAL — SELEÇÃO DE ATRIBUTOS SOBRE COMBINAÇÕES")
print(f"{'='*70}")
resumo = pd.DataFrame(resultados_globais)
print(resumo.to_string(index=False))
resumo.to_csv('resultado_selecao_atributos_combinacoes.csv', index=False)
print("\n✅ Salvo: resultado_selecao_atributos_combinacoes.csv")

print("""
=============================================================
COMO INTERPRETAR PRA APRESENTAÇÃO:
=============================================================
- Comparar cada linha "TODOS os atributos" com as linhas de
  Ranker/RFE da MESMA combinação. Se a seleção igualar ou
  superar o baseline da combinação, prova que parte da
  concatenação era ruído/redundância.

- Comparar também contra os tiers isolados do script 12:
  Tier 1 Reforçado (sozinho):           57,64% | Kappa 0,317
  Tier 2 Reforçado (sozinho):           57,13% | Kappa 0,305
  Tier 3 BERT (sozinho, RF):            58,73% | Kappa 0,333
  Tier1+Tier2 (script 12):              57,68% | Kappa 0,3186
  Tier1+Tier2+Tema (script 12):         57,72% | Kappa 0,3177
  Tier1+Tier3 PCA (script 12):          58,45% | Kappa 0,3254

- Se mesmo com seleção de atributos nenhuma combinação superar
  claramente o Tier 3 isolado, isso REFORÇA a conclusão honesta:
  "concatenar features de níveis diferentes não agrega valor
  significativo neste problema — o BERT já captura a maior
  parte do sinal disponível, e features de superfície/gramática
  trazem pouca informação complementar". É um resultado válido
  para discussão, mesmo sendo negativo.
=============================================================
""")
