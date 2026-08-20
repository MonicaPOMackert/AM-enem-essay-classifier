"""
=============================================================
COMBINAÇÕES DE TIERS — Yuri
=============================================================
Pedido do professor: "mais de tudo". Até agora testamos cada
Tier ISOLADAMENTE (Tier 1, Tier 2, Tier 3). Esse script testa
COMBINAÇÕES de Tiers — juntando as features de diferentes
níveis para ver se a soma supera as partes individuais.

Combinações testadas:
 1) Tier 1 Reforçado + Tier 2 Reforçado (superfície + gramática)
 2) Tier 1 Reforçado + Tier 2 Reforçado + Feature de Tema (combo
    interpretável completo)
 3) Tier 1 Reforçado + Tier 3 BERT (interpretável + caixa-preta)

PRÉ-REQUISITO: ter na mesma pasta:
 - tier1_reforcado_500.csv
 - tier2_reforcado.csv
 - feature_tema.csv
 - redacoes_tier3_bert.csv (ou redacoes_tier3_bert_processado.csv)

ANTES DE RODAR:
 pip install pandas numpy scikit-learn
=============================================================
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, precision_recall_fscore_support,
    matthews_corrcoef
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
ORDEM_CLASSES = ['Insuficiente', 'Regular', 'Bom', 'Excelente']

def mcc_por_classe(y_true, y_pred, classes):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    return {c: matthews_corrcoef((y_true == c).astype(int), (y_pred == c).astype(int)) for c in classes}

def avaliar(nome, X, y):
    print(f"\n{'-'*70}")
    print(f"{nome}  (n_atributos={X.shape[1]})")
    y_pred = cross_val_predict(rf, X, y, cv=cv)
    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    mcc_geral = matthews_corrcoef(y, y_pred)
    print(f"Acurácia: {acc*100:.2f}%  |  Kappa: {kappa:.4f}  |  MCC: {mcc_geral:.4f}")

    precisao, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, labels=ORDEM_CLASSES, zero_division=0
    )
    mcc_c = mcc_por_classe(y, y_pred, ORDEM_CLASSES)
    tabela = pd.DataFrame({
        'Classe': ORDEM_CLASSES, 'Recall': np.round(recall,3),
        'Precision': np.round(precisao,3), 'MCC': [round(mcc_c[c],3) for c in ORDEM_CLASSES]
    })
    print(tabela.to_string(index=False))

    return {'experimento': nome, 'n_atributos': X.shape[1],
            'acuracia': round(acc*100,2), 'kappa': round(kappa,4), 'mcc_geral': round(mcc_geral,4)}

resultados = []

print("Carregando arquivos...")
tier1 = pd.read_csv('tier1_reforcado_500.csv')
tier2 = pd.read_csv('tier2_reforcado.csv')
tema = pd.read_csv('feature_tema.csv')
print(f"Tier 1: {tier1.shape}  |  Tier 2: {tier2.shape}  |  Tema: {tema.shape}")

y = tier1['score_class']  # mesma ordem de linhas em todos os arquivos

# ── 1. Tier 1 + Tier 2 (sem duplicar score_class) ──────────────
X_t1 = tier1.drop(columns=['score_class'])
X_t2 = tier2.drop(columns=['score_class'])
X_combo12 = pd.concat([X_t1, X_t2], axis=1).values
resultados.append(avaliar("Tier 1 Reforçado + Tier 2 Reforçado", X_combo12, y))

# ── 2. Tier 1 + Tier 2 + Tema ───────────────────────────────────
X_combo12_tema = pd.concat([X_t1, X_t2, tema[['feature_aderencia_tema']]], axis=1).values
resultados.append(avaliar("Tier 1 + Tier 2 + Tema (combo interpretável completo)", X_combo12_tema, y))

# ── 3. Tier 1 + Tier 3 BERT (se o arquivo existir) ──────────────
try:
    tier3 = pd.read_csv('redacoes_tier3_bert.csv')
    print(f"\nTier 3 carregado: {tier3.shape}")
    X_t3 = tier3.drop(columns=['score_class'])

    # Reduz Tier 3 via PCA antes de combinar (768 colunas é demais para somar direto)
    scaler = StandardScaler()
    X_t3_scaled = scaler.fit_transform(X_t3.values)
    pca = PCA(n_components=0.95, random_state=42)
    X_t3_pca = pca.fit_transform(X_t3_scaled)
    print(f"Tier 3 após PCA: {X_t3_pca.shape}")

    X_combo_t1_t3 = np.hstack([X_t1.values, X_t3_pca])
    resultados.append(avaliar("Tier 1 Reforçado + Tier 3 BERT (PCA)", X_combo_t1_t3, y))
except FileNotFoundError:
    print("\n⚠️  redacoes_tier3_bert.csv não encontrado — pulando combinação com Tier 3")

# ── Tabela final ────────────────────────────────────────────────
print(f"\n{'='*70}")
print("TABELA COMPARATIVA — COMBINAÇÕES DE TIERS")
print(f"{'='*70}")
resumo = pd.DataFrame(resultados)
print(resumo.to_string(index=False))
resumo.to_csv('resultado_combinacoes_tiers.csv', index=False)
print("\n✅ Salvo: resultado_combinacoes_tiers.csv")

print("""
=============================================================
COMPARAR COM OS TIERS ISOLADOS (já temos):
=============================================================
Tier 1 Reforçado (sozinho):              57,64%  |  Kappa 0,317
Tier 2 Reforçado (sozinho):              57,13%  |  Kappa 0,305
Tier 3 BERT (sozinho, Random Forest):    58,73%  |  Kappa 0,333

Se a COMBINAÇÃO Tier 1+2 superar os dois isolados, mostra que
as features de diferentes níveis se complementam (uma capta
o que a outra não capta).

Se Tier 1+Tier 3 (interpretável + caixa-preta) superar o Tier 3
isolado, é um resultado muito interessante: significa que mesmo
o BERT se beneficia de informação estrutural simples que ele
não captura por si só.
=============================================================
""")
