"""
=============================================================
CALIBRAÇÃO DO DBSCAN — testando vários valores de eps
=============================================================
Esse script é só um complemento do script 1 (clusterização).
Roda RÁPIDO porque só testa o DBSCAN com vários valores de eps,
pra achar um que gere uma divisão razoável (nem 1 cluster só,
nem tudo classificado como ruído).

PRÉ-REQUISITO: o mesmo redacoes_tier3_bert.csv usado no script 1.

ANTES DE RODAR:
 pip install pandas numpy scikit-learn
=============================================================
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

CSV_PATH = 'redacoes_tier3_bert.csv'   # <-- ajuste o caminho

print("Carregando dados...")
df = pd.read_csv(CSV_PATH)
y_true = df['score_class']
X = df.drop(columns=['score_class']).values

print("Normalizando e aplicando PCA (95% variância)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Shape após PCA: {X_pca.shape}\n")

# ── Testa vários valores de eps ────────────────────────────────
valores_eps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

print(f"{'eps':>6} | {'n_clusters':>10} | {'n_ruido':>8} | {'% ruido':>8} | {'maior cluster':>14}")
print("-" * 65)

for eps in valores_eps:
    dbscan = DBSCAN(eps=eps, min_samples=10)
    labels = dbscan.fit_predict(X_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_ruido = list(labels).count(-1)
    pct_ruido = 100 * n_ruido / len(labels)

    # Tamanho do maior cluster (excluindo ruído)
    if n_clusters > 0:
        tamanhos = pd.Series(labels[labels != -1]).value_counts()
        maior_cluster = tamanhos.max()
        pct_maior = 100 * maior_cluster / len(labels)
        info_maior = f"{maior_cluster} ({pct_maior:.1f}%)"
    else:
        info_maior = "—"

    print(f"{eps:>6.1f} | {n_clusters:>10} | {n_ruido:>8} | {pct_ruido:>7.1f}% | {info_maior:>14}")

print("""
=============================================================
COMO ESCOLHER O MELHOR eps:
=============================================================
- EVITE eps onde n_clusters = 1 e o "maior cluster" é ~100%
  (significa que tudo caiu num cluster só — eps grande demais)
- EVITE eps onde % ruido é muito alto, tipo > 80%
  (significa que quase nada formou cluster — eps pequeno demais)
- PROCURE um eps que dê entre 2 e 8 clusters, com ruído
  controlado (idealmente abaixo de 30-40%)

Depois de escolher o eps, volte pro script 1 (clusterizacao_MONICA.py)
e troque o valor de eps na linha do DBSCAN, rode de novo o script
completo (vai gerar ARI, NMI, Homogeneidade e Completude corretos
pra esse eps calibrado).
=============================================================
""")
