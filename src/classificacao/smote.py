"""
=============================================================
BALANCEAMENTO COM SMOTE — Danilo
=============================================================
Conecta diretamente com o achado mais forte do projeto: no
cenário BINÁRIO, o Random Forest teve Acurácia de 88,91% mas
Kappa de apenas 0,038 — ele "chutava Aprovado" quase sempre
porque a base é desbalanceada (89% Aprovado vs 11% Insuficiente).

Esse script testa se o SMOTE (Synthetic Minority Oversampling
Technique) resolve esse problema. O SMOTE cria exemplos
SINTÉTICOS da classe minoritária (Insuficiente) interpolando
entre exemplos reais existentes, balanceando a base ANTES do
treino.

Testamos SMOTE tanto no cenário BINÁRIO quanto no MULTICLASSE,
comparando com os resultados SEM balanceamento que já temos.

NÃO precisa gerar o BERT de novo — usa o mesmo CSV de antes.

ANTES DE RODAR:
 pip install pandas numpy scikit-learn imbalanced-learn
=============================================================
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, precision_recall_fscore_support,
    matthews_corrcoef
)
from imblearn.over_sampling import SMOTE

# ── AJUSTE AQUI O CAMINHO DO SEU ARQUIVO ─────────────────────
CSV_PATH = 'redacoes_tier3_bert.csv'   # <-- coloque o caminho real no seu PC

print("Carregando dados do Tier 3 (BERTimbau)...")
df = pd.read_csv(CSV_PATH)
X = df.drop(columns=['score_class']).values
y_multiclasse = df['score_class']

def binarizar(classe):
    return 'Insuficiente' if classe == 'Insuficiente' else 'Aprovado'
y_binario = y_multiclasse.apply(binarizar)

print("Normalizando e aplicando PCA (95% variância)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Shape após PCA: {X_pca.shape}")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

modelos = {
    'SMO / SVM': SVC(C=1.0, kernel='linear', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'J48 / Decision Tree': DecisionTreeClassifier(min_samples_leaf=10, random_state=42),
}

def avaliar_com_smote(nome_cenario, X_data, y_data, classes_ordem):
    print(f"\n{'='*70}")
    print(f"CENÁRIO: {nome_cenario} — COM SMOTE")
    print(f"{'='*70}")
    resultados_locais = []

    for nome_modelo, modelo in modelos.items():
        y_preds, y_trues = [], []

        for train_idx, test_idx in cv.split(X_data, y_data):
            X_train, X_test = X_data[train_idx], X_data[test_idx]
            y_train = y_data.iloc[train_idx] if hasattr(y_data, 'iloc') else y_data[train_idx]
            y_test = y_data.iloc[test_idx] if hasattr(y_data, 'iloc') else y_data[test_idx]

            # SMOTE aplicado SÓ no treino (nunca no teste, para não "vazar" informação)
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

            modelo.fit(X_train_bal, y_train_bal)
            y_pred = modelo.predict(X_test)

            y_preds.extend(y_pred)
            y_trues.extend(y_test)

        acc = accuracy_score(y_trues, y_preds)
        kappa = cohen_kappa_score(y_trues, y_preds)
        mcc_geral = matthews_corrcoef(y_trues, y_preds)

        precisao, recall, f1, _ = precision_recall_fscore_support(
            y_trues, y_preds, labels=classes_ordem, zero_division=0
        )

        print(f"\n{nome_modelo}")
        print(f"Acurácia: {acc*100:.2f}%  |  Kappa: {kappa:.4f}  |  MCC: {mcc_geral:.4f}")
        for i, c in enumerate(classes_ordem):
            print(f"  {c:15s} Recall: {recall[i]:.3f}  Precision: {precisao[i]:.3f}")

        idx_insuf = classes_ordem.index('Insuficiente')
        resultados_locais.append({
            'cenario': nome_cenario, 'modelo': nome_modelo,
            'acuracia': round(acc*100,2), 'kappa': round(kappa,4),
            'mcc_geral': round(mcc_geral,4),
            'recall_insuficiente': round(recall[idx_insuf],4),
        })
    return resultados_locais

resultados = []
resultados += avaliar_com_smote("MULTICLASSE + SMOTE", X_pca, y_multiclasse, ['Insuficiente','Regular','Bom','Excelente'])
resultados += avaliar_com_smote("BINÁRIO + SMOTE", X_pca, y_binario, ['Insuficiente','Aprovado'])

print(f"\n{'='*70}")
print("TABELA COMPARATIVA FINAL — COM SMOTE")
print(f"{'='*70}")
resumo = pd.DataFrame(resultados)
print(resumo.to_string(index=False))
resumo.to_csv('resultado_smote.csv', index=False)
print("\n✅ Salvo: resultado_smote.csv")

print("""
=============================================================
COMPARAR COM O RESULTADO SEM SMOTE (já temos):
=============================================================
BINÁRIO SEM SMOTE:
  SMO:           90,07% acc | Kappa 0,343 | Recall Insuf: 28,0%
  Random Forest: 88,91% acc | Kappa 0,038 | Recall Insuf: 2,3%  <- problema
  J48:           85,69% acc | Kappa 0,191 | Recall Insuf: 23,4%

O que esperar com SMOTE:
- A Acurácia geral pode CAIR um pouco (é normal e esperado)
- O Recall de Insuficiente deve SUBIR bastante, especialmente
  no Random Forest — é a prova de que o SMOTE força o modelo
  a aprender padrões da classe minoritária, em vez de ignorá-la

NARRATIVA PARA APRESENTAÇÃO:
"O SMOTE trocou uma Acurácia inflada por um modelo genuinamente
mais balanceado entre as classes — uma troca que vale a pena
quando o objetivo é identificar corretamente redações Insuficientes,
não só maximizar o número de acertos totais."
=============================================================
""")
