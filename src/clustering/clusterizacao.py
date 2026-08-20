"""
=============================================================
CLUSTERIZAÇÃO — KMeans, EM (Gaussian Mixture) e DBSCAN
=============================================================
O que esse script faz:
 1. Carrega o Tier 3 (BERTimbau) já pronto
 2. Reduz dimensionalidade com PCA (igual fizemos no Weka, 95% variância)
 3. Roda os 3 algoritmos de clustering SEM usar a coluna de nota
 4. Compara os clusters formados com as classes reais (Insuficiente/Regular/Bom/Excelente)
 5. Gera uma tabela de contingência (cluster x classe) pra discutirmos

NÃO precisa gerar o BERT de novo — usa o CSV que o Yuri já gerou.

ANTES DE RODAR:
 pip install pandas numpy scikit-learn
=============================================================
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, homogeneity_score, completeness_score
)

# ── AJUSTE AQUI O CAMINHO DO SEU ARQUIVO ─────────────────────
CSV_PATH = 'redacoes_tier3_bert.csv'   # <-- coloque o caminho real no seu PC

# ── Carrega dados ─────────────────────────────────────────────
print("Carregando dados do Tier 3 (BERTimbau)...")
df = pd.read_csv(CSV_PATH)

y_true = df['score_class']                      # classes reais (não usamos no treino)
X = df.drop(columns=['score_class']).values      # 768 colunas do BERT

print(f"Shape original: {X.shape}")

# ── Normalização + PCA (mesmo procedimento do Weka: 95% variância) ──
print("Normalizando e aplicando PCA (95% variância)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Shape após PCA: {X_pca.shape} (mantendo 95% da variância)")

# ── Função auxiliar para avaliar e imprimir resultados ────────
def avaliar_cluster(nome, labels, X_data):
    # Remove ruído do DBSCAN (label -1) da contagem de silhueta, se necessário
    mask = labels != -2  # placeholder, não filtra nada por padrão
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_ruido = list(labels).count(-1)

    print(f"\n{'='*60}")
    print(f"RESULTADO — {nome}")
    print(f"{'='*60}")
    print(f"Número de clusters encontrados: {n_clusters}")
    if n_ruido > 0:
        print(f"Pontos classificados como ruído: {n_ruido} ({100*n_ruido/len(labels):.1f}%)")

    # Métricas de comparação com classes reais (não usadas no treino, só para análise)
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    homog = homogeneity_score(y_true, labels)
    complet = completeness_score(y_true, labels)

    print(f"Adjusted Rand Index (ARI):     {ari:.4f}  (1.0 = match perfeito com as classes, 0 = aleatório)")
    print(f"Normalized Mutual Info (NMI):  {nmi:.4f}  (quanta informação os clusters compartilham com as classes)")
    print(f"Homogeneidade:                 {homog:.4f}  (cada cluster contém só 1 classe?)")
    print(f"Completude:                    {complet:.4f}  (cada classe está num único cluster?)")

    if n_clusters > 1:
        try:
            sil = silhouette_score(X_data, labels)
            print(f"Silhouette Score:              {sil:.4f}  (qualidade da separação dos clusters, -1 a 1)")
        except Exception:
            pass

    # Tabela de contingência: cluster x classe real
    print("\nTabela de Contingência (Cluster x Classe Real):")
    tabela = pd.crosstab(labels, y_true, rownames=['Cluster'], colnames=['Classe Real'])
    print(tabela)
    return {'nome': nome, 'ari': ari, 'nmi': nmi, 'homogeneidade': homog, 'completude': complet}

resultados = []

# ── 1. KMeans (k=4, igual ao número de classes) ───────────────
print("\nRodando KMeans (k=4)...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_pca)
resultados.append(avaliar_cluster("KMeans (k=4)", labels_kmeans, X_pca))

# ── 2. EM / Gaussian Mixture (k=4) ─────────────────────────────
print("\nRodando EM (Gaussian Mixture, k=4)...")
gmm = GaussianMixture(n_components=4, random_state=42)
labels_gmm = gmm.fit_predict(X_pca)
resultados.append(avaliar_cluster("EM - Gaussian Mixture (k=4)", labels_gmm, X_pca))

# ── 3. DBSCAN (precisa testar eps manualmente) ─────────────────
print("\nRodando DBSCAN...")
# eps e min_samples podem precisar de ajuste — comece com esses valores
# Se der "1 cluster só" ou "tudo ruído", ajuste eps (tente 0.5, 1, 2, 3, 5...)
dbscan = DBSCAN(eps=19.90, min_samples=10)
labels_dbscan = dbscan.fit_predict(X_pca)
resultados.append(avaliar_cluster("DBSCAN (eps=19.90, min_samples=10)", labels_dbscan, X_pca))

# ── Resumo final comparando os 3 ───────────────────────────────
print(f"\n{'='*60}")
print("RESUMO COMPARATIVO")
print(f"{'='*60}")
resumo_df = pd.DataFrame(resultados)
print(resumo_df.to_string(index=False))

resumo_df.to_csv('resultado_clusterizacao.csv', index=False)
print("\nResumo salvo em: resultado_clusterizacao.csv")

print("""
=============================================================
COMO INTERPRETAR PRA APRESENTAÇÃO:
=============================================================
- Se ARI e NMI forem baixos (próximos de 0): os clusters NÃO
  batem com as categorias de nota. Isso é esperado e até
  interessante de discutir — significa que a similaridade
  textual "natural" das redações não segue as faixas de nota
  do ENEM, que são definidas por critérios de correção, não
  por similaridade de conteúdo.

- DBSCAN pode jogar muita coisa como "ruído" (-1) se o eps
  estiver mal calibrado — teste alguns valores diferentes de
  eps até achar uma divisão razoável (2 a 6 clusters).

- Vale tentar também k=2 e k=3 no KMeans/EM pra ver se aparece
  algum agrupamento natural diferente das 4 classes oficiais.
=============================================================
""")
