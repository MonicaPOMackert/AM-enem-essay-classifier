"""
=============================================================
MAIS ALGORITMOS DE CLASSIFICAÇÃO — Danilo
=============================================================
Pedido do professor: "vale a utilização de mais algoritmos até
o final, e as próximas técnicas que vamos ver também."

Esse script testa 4 algoritmos NOVOS que ainda não tínhamos
usado no projeto, todos sobre o Tier 3 (BERTimbau + PCA), para
comparar diretamente com os 8 que já testamos na parcial:

 1) MLPClassifier (Multilayer Perceptron) — rede neural simples,
    equivalente ao MultilayerPerceptron do Weka
 2) LogisticRegression — modelo linear probabilístico, equivalente
    ao SimpleLogistic do Weka
 3) GradientBoostingClassifier — ensemble sequencial (cada árvore
    corrige o erro da anterior), diferente do Random Forest
    (que treina árvores em paralelo/independentes)
 4) VotingClassifier — combina os 3 melhores modelos já testados
    no projeto (SMO, Random Forest, MLP) por votação majoritária

NÃO precisa gerar o BERT de novo — usa o mesmo CSV de antes.

ANTES DE RODAR:
 pip install pandas numpy scikit-learn
=============================================================
"""
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    precision_recall_fscore_support, matthews_corrcoef
)

# ── AJUSTE AQUI O CAMINHO DO SEU ARQUIVO ─────────────────────
CSV_PATH = 'redacoes_tier3_bert.csv'   # <-- coloque o caminho real no seu PC

print("Carregando dados do Tier 3 (BERTimbau)...")
df = pd.read_csv(CSV_PATH)
X = df.drop(columns=['score_class']).values
y = df['score_class']

print("Normalizando e aplicando PCA (95% variância)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Shape após PCA: {X_pca.shape}")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
ORDEM_CLASSES = ['Insuficiente', 'Regular', 'Bom', 'Excelente']

def mcc_por_classe(y_true, y_pred, classes):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    return {c: matthews_corrcoef((y_true == c).astype(int), (y_pred == c).astype(int)) for c in classes}

def avaliar(nome, modelo, X_data, y_data):
    print(f"\n{'-'*70}")
    print(f"Rodando: {nome}  (pode demorar alguns minutos)")
    inicio = time.time()
    y_pred = cross_val_predict(modelo, X_data, y_data, cv=cv)
    duracao = time.time() - inicio

    acc = accuracy_score(y_data, y_pred)
    kappa = cohen_kappa_score(y_data, y_pred)
    mcc_geral = matthews_corrcoef(y_data, y_pred)

    print(f"{nome}")
    print(f"Acurácia: {acc*100:.2f}%  |  Kappa: {kappa:.4f}  |  MCC: {mcc_geral:.4f}  |  Tempo: {duracao:.1f}s")

    precisao, recall, f1, suporte = precision_recall_fscore_support(
        y_data, y_pred, labels=ORDEM_CLASSES, zero_division=0
    )
    mcc_classes = mcc_por_classe(y_data, y_pred, ORDEM_CLASSES)
    tabela = pd.DataFrame({
        'Classe': ORDEM_CLASSES, 'Recall': np.round(recall,3),
        'Precision': np.round(precisao,3), 'F1': np.round(f1,3),
        'MCC': [round(mcc_classes[c],3) for c in ORDEM_CLASSES]
    })
    print(tabela.to_string(index=False))

    return {
        'modelo': nome, 'acuracia': round(acc*100,2), 'kappa': round(kappa,4),
        'mcc_geral': round(mcc_geral,4), 'tempo_seg': round(duracao,1)
    }

resultados = []

# ── 1. MLP (Multilayer Perceptron) ─────────────────────────────
mlp = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=300, random_state=42)
resultados.append(avaliar("MLPClassifier (rede neural, 2 camadas)", mlp, X_pca, y))

# ── 2. Logistic Regression ──────────────────────────────────────
logreg = LogisticRegression(max_iter=1000, random_state=42)
resultados.append(avaliar("LogisticRegression (multinomial)", logreg, X_pca, y))

# ── 3. Gradient Boosting ────────────────────────────────────────
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
resultados.append(avaliar("GradientBoostingClassifier (100 estimators)", gb, X_pca, y))

# ── 4. Voting Classifier (combina os melhores) ──────────────────
voting = VotingClassifier(estimators=[
    ('svm', SVC(C=1.0, kernel='linear', probability=True, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,50), max_iter=300, random_state=42)),
], voting='soft')
resultados.append(avaliar("VotingClassifier (SMO + RF + MLP, soft voting)", voting, X_pca, y))

# ── Tabela final ────────────────────────────────────────────────
print(f"\n{'='*70}")
print("TABELA COMPARATIVA — NOVOS ALGORITMOS")
print(f"{'='*70}")
resumo = pd.DataFrame(resultados)
print(resumo.to_string(index=False))
resumo.to_csv('resultado_novos_algoritmos.csv', index=False)
print("\n✅ Salvo: resultado_novos_algoritmos.csv")

print("""
=============================================================
COMPARAR COM O QUE JÁ TÍNHAMOS (apresentação parcial/final):
=============================================================
SMO C=1.0 (campeão atual):        60,37%  |  Kappa 0,386
Random Forest 300 árvores:        57,24%  |  Kappa 0,300
J48 (minNumObj=50):                50,70%  |  Kappa 0,222

Se o VotingClassifier (que combina SMO+RF+MLP) superar o SMO
isolado, é um ótimo resultado para apresentação: mostra que
combinar modelos diferentes captura mais nuances que um só.

Se o MLP ou GradientBoosting ficarem próximos do SMO, vale
destacar como "técnicas modernas alcançando resultado competitivo".
=============================================================
""")
