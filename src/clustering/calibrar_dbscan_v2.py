"""
=============================================================
CALIBRAÇÃO AUTOMÁTICA DO eps — usando k-distance graph
=============================================================
A primeira tentativa de calibração testou eps de 0.5 a 10.0 e
TODOS deram 100% de ruído — ou seja, a escala real das distâncias
nos seus dados (após PCA) é maior do que isso.

Esse script calcula automaticamente a distância média ao 10º
vizinho mais próximo de cada ponto (k-distance, técnica padrão
pra calibrar DBSCAN) e usa isso pra sugerir uma faixa de eps
muito mais precisa, na escala correta dos SEUS dados.

PRÉ-REQUISITO: o mesmo redacoes_tier3_bert.csv usado antes.

ANTES DE RODAR:
 pip install pandas numpy scikit-learn
=============================================================
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

CSV_PATH = r'redacoes_tier3_bert.csv'   # <-- ajuste o caminho se necessário

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

# ═══════════════════════════════════════════════════════════
# PASSO 1 — Calcula a distância ao k-ésimo vizinho mais próximo
# (k = min_samples, técnica padrão pra calibrar eps no DBSCAN)
# ═══════════════════════════════════════════════════════════
MIN_SAMPLES = 10
print(f"Calculando distância ao {MIN_SAMPLES}º vizinho mais próximo de cada ponto...")

vizinhos = NearestNeighbors(n_neighbors=MIN_SAMPLES)
vizinhos.fit(X_pca)
distancias, _ = vizinhos.kneighbors(X_pca)

# A distância ao k-ésimo vizinho (última coluna) é a métrica relevante
k_distancias = np.sort(distancias[:, -1])

print(f"\nEstatísticas da distância ao {MIN_SAMPLES}º vizinho:")
print(f"  Mínimo:        {k_distancias.min():.2f}")
print(f"  Percentil 25%: {np.percentile(k_distancias, 25):.2f}")
print(f"  Mediana:       {np.percentile(k_distancias, 50):.2f}")
print(f"  Percentil 75%: {np.percentile(k_distancias, 75):.2f}")
print(f"  Percentil 90%: {np.percentile(k_distancias, 90):.2f}")
print(f"  Máximo:        {k_distancias.max():.2f}")

# ═══════════════════════════════════════════════════════════
# PASSO 2 — Sugere valores de eps na escala correta e testa
# ═══════════════════════════════════════════════════════════
# Valores sugeridos: do percentil 10 ao percentil 90 da distância
p10 = np.percentile(k_distancias, 10)
p90 = np.percentile(k_distancias, 90)
valores_eps_sugeridos = np.linspace(p10, p90, 10)

print(f"\nValores de eps sugeridos (baseados na escala real dos dados):")
print([round(v, 2) for v in valores_eps_sugeridos])

print(f"\n{'eps':>8} | {'n_clusters':>10} | {'n_ruido':>8} | {'% ruido':>8} | {'maior cluster':>14}")
print("-" * 70)

melhor_eps = None
melhor_score = -1

for eps in valores_eps_sugeridos:
    dbscan = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
    labels = dbscan.fit_predict(X_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_ruido = list(labels).count(-1)
    pct_ruido = 100 * n_ruido / len(labels)

    if n_clusters > 0:
        tamanhos = pd.Series(labels[labels != -1]).value_counts()
        maior_cluster = tamanhos.max()
        pct_maior = 100 * maior_cluster / len(labels)
        info_maior = f"{maior_cluster} ({pct_maior:.1f}%)"

        # Heurística simples pra sugerir o "melhor": entre 2-8 clusters,
        # ruído moderado, sem 1 cluster dominante > 90%
        if 2 <= n_clusters <= 8 and pct_ruido < 50 and pct_maior < 90:
            score = n_clusters - (pct_ruido / 100)  # prefere mais clusters, menos ruído
            if score > melhor_score:
                melhor_score = score
                melhor_eps = eps
    else:
        info_maior = "—"

    print(f"{eps:>8.2f} | {n_clusters:>10} | {n_ruido:>8} | {pct_ruido:>7.1f}% | {info_maior:>14}")

print("\n" + "="*70)
if melhor_eps is not None:
    print(f"✅ SUGESTÃO AUTOMÁTICA: eps = {melhor_eps:.2f}")
    print(f"   Use esse valor no script 1_clusterizacao_MONICA.py")
else:
    print("⚠️  Nenhum valor testado ficou numa faixa ideal automaticamente.")
    print("   Olhe a tabela acima e escolha manualmente um eps que tenha:")
    print("   - Entre 2 e 8 clusters")
    print("   - Ruído abaixo de 50%")
    print("   - Nenhum cluster dominando mais de 90% dos pontos")
print("="*70)
