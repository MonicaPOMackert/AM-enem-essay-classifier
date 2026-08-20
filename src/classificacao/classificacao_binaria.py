"""
=============================================================
CLASSIFICAÇÃO BINÁRIA v2 — Com métricas completas por classe
=============================================================
EVOLUÇÃO da versão anterior: o script antigo só calculava
Acurácia e Kappa GERAL (média entre as classes). Essa versão
adiciona Recall, Precision, F1 e MCC POR CLASSE, pra confirmar
uma suspeita importante:

No resultado anterior, o Random Forest no cenário BINÁRIO teve
Acurácia de 88,91% mas Kappa de apenas 0,038 — isso é sinal de
um problema clássico em bases desbalanceadas: o modelo pode estar
"chutando Aprovado" quase sempre (já que 88,75% da base é
Aprovado) sem realmente aprender a identificar Insuficiente.

Esse script confirma ou refuta essa suspeita mostrando o Recall
específico da classe "Insuficiente" — se for baixo (tipo < 20%),
confirma que o modelo é ruim em identificar quem reprovou, mesmo
com Acurácia geral alta.

NÃO precisa gerar o BERT de novo — usa o mesmo CSV de antes.

ANTES DE RODAR:
 pip install pandas numpy scikit-learn
=============================================================
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    precision_recall_fscore_support, matthews_corrcoef
)

# ── AJUSTE AQUI O CAMINHO DO SEU ARQUIVO ─────────────────────
CSV_PATH = 'redacoes_tier3_bert.csv'   # <-- coloque o caminho real no seu PC

# ── Carrega dados ─────────────────────────────────────────────
print("Carregando dados do Tier 3 (BERTimbau)...")
df = pd.read_csv(CSV_PATH)

X = df.drop(columns=['score_class']).values
y_multiclasse = df['score_class']

def binarizar(classe):
    return 'Insuficiente' if classe == 'Insuficiente' else 'Aprovado'

y_binario = y_multiclasse.apply(binarizar)

print("\nDistribuição BINÁRIA:")
print(y_binario.value_counts())
print(f"Proporção: {y_binario.value_counts(normalize=True).round(4).to_dict()}")

print("\nNormalizando e aplicando PCA (95% variância)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Shape após PCA: {X_pca.shape}")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

modelos = {
    'SMO / SVM (C=1.0)': SVC(C=1.0, kernel='linear', random_state=42),
    'Random Forest (100 árvores)': RandomForestClassifier(n_estimators=100, random_state=42),
    'J48 / Decision Tree': DecisionTreeClassifier(min_samples_leaf=10, random_state=42),
}

ORDEM_CLASSES_MULTI = ['Insuficiente', 'Regular', 'Bom', 'Excelente']
ORDEM_CLASSES_BIN = ['Insuficiente', 'Aprovado']

def mcc_por_classe(y_true, y_pred, classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    resultado = {}
    for classe in classes:
        y_true_bin = (y_true == classe).astype(int)
        y_pred_bin = (y_pred == classe).astype(int)
        resultado[classe] = matthews_corrcoef(y_true_bin, y_pred_bin)
    return resultado

def avaliar_completo(nome_cenario, nome_modelo, modelo, X_data, y, classes):
    y_pred = cross_val_predict(modelo, X_data, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    mcc_geral = matthews_corrcoef(y, y_pred)

    precisao, recall, f1, suporte = precision_recall_fscore_support(
        y, y_pred, labels=classes, zero_division=0
    )
    mcc_classes = mcc_por_classe(y, y_pred, classes)

    print(f"\n{'-'*70}")
    print(f"{nome_cenario} — {nome_modelo}")
    print(f"{'-'*70}")
    print(f"Acurácia: {acc*100:.2f}%  |  Kappa: {kappa:.4f}  |  MCC geral: {mcc_geral:.4f}")

    tabela = pd.DataFrame({
        'Classe': classes,
        'Recall': np.round(recall, 3),
        'Precision': np.round(precisao, 3),
        'F1-Score': np.round(f1, 3),
        'MCC': [round(mcc_classes[c], 3) for c in classes],
        'Suporte (n)': suporte
    })
    print(tabela.to_string(index=False))

    cm = confusion_matrix(y, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"real_{c}" for c in classes],
                          columns=[f"prev_{c}" for c in classes])
    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    linha_resumo = {
        'cenario': nome_cenario, 'modelo': nome_modelo,
        'acuracia': round(acc*100, 2), 'kappa': round(kappa, 4),
        'mcc_geral': round(mcc_geral, 4),
    }
    # Adiciona recall/precision/mcc específico da classe "Insuficiente"
    # (a classe minoritária, a mais importante de verificar)
    idx_insuf = classes.index('Insuficiente')
    linha_resumo['recall_insuficiente'] = round(recall[idx_insuf], 4)
    linha_resumo['precision_insuficiente'] = round(precisao[idx_insuf], 4)
    linha_resumo['mcc_insuficiente'] = round(mcc_classes['Insuficiente'], 4)

    return linha_resumo

resultados = []

print(f"\n{'='*70}")
print("CENÁRIO MULTICLASSE (4 classes)")
print(f"{'='*70}")
for nome_modelo, modelo in modelos.items():
    resultados.append(avaliar_completo(
        "MULTICLASSE", nome_modelo, modelo, X_pca, y_multiclasse, ORDEM_CLASSES_MULTI
    ))

print(f"\n{'='*70}")
print("CENÁRIO BINÁRIO (Aprovado vs Insuficiente)")
print(f"{'='*70}")
for nome_modelo, modelo in modelos.items():
    resultados.append(avaliar_completo(
        "BINÁRIO", nome_modelo, modelo, X_pca, y_binario, ORDEM_CLASSES_BIN
    ))

print(f"\n{'='*70}")
print("TABELA COMPARATIVA FINAL")
print(f"{'='*70}")
resumo = pd.DataFrame(resultados)
print(resumo.to_string(index=False))
resumo.to_csv('resultado_binario_vs_multiclasse_v2.csv', index=False)
print("\n✅ Salvo: resultado_binario_vs_multiclasse_v2.csv")

print("""
=============================================================
O QUE VERIFICAR NESSE RESULTADO:
=============================================================
Compare a coluna 'recall_insuficiente' entre os modelos no
cenário BINÁRIO:

- Se o Random Forest tiver recall_insuficiente MUITO baixo
  (tipo < 0.20) junto com Kappa baixo (já vimos 0.038 antes),
  isso CONFIRMA que o modelo está "chutando Aprovado" quase
  sempre e raramente identifica um Insuficiente de verdade —
  apesar da Acurácia geral parecer ótima (88,91%).

- Compare com o SMO: se o recall_insuficiente do SMO for mais
  alto que o do Random Forest, mesmo com Acurácia total um
  pouco menor, isso mostra que o SMO é o modelo mais "justo"
  entre as classes, não só o que "acerta mais no total".

Esse é o ponto-chave pra apresentação: ACURÁCIA ALTA NÃO
SIGNIFICA MODELO BOM quando a base é desbalanceada.
=============================================================
""")
