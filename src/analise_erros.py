"""
=============================================================
ANÁLISE DE ERROS — Mônica
=============================================================
Pedido do professor: "mais de tudo". Até agora só reportamos
MÉTRICAS AGREGADAS (acurácia, kappa, recall por classe). Esse
script vai além: investiga QUAIS redações específicas o melhor
modelo (SMO no Tier 3) mais erra, e se existe um PADRÃO nesses
erros.

Perguntas que esse script responde:
 1) O modelo erra mais em redações curtas, longas, ou não importa?
 2) Existe relação entre o erro e a "distância" da nota para o
    limite entre classes (ex: nota 395 é quase Regular, mas é
    Insuficiente — são esses os mais difíceis?)
 3) Quais são as redações "mais difíceis" (erradas por quase
    todos os modelos), e quais são "fáceis" (todos acertam)?

PRÉ-REQUISITO: ter na mesma pasta:
 - redacoes_tier3_bert.csv (ou redacoes_tier3_bert_processado.csv)
 - essay-br(4570TEXTOS) (2).csv (para recuperar o texto e a nota
   numérica original das redações, não só a classe)

ANTES DE RODAR:
 pip install pandas numpy scikit-learn
=============================================================
"""
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ── AJUSTE AQUI OS CAMINHOS ──────────────────────────────────
CSV_BERT = 'redacoes_tier3_bert.csv'
CSV_ORIGINAL = 'essay-br(4570TEXTOS) (2).csv'

def parse_essay(raw):
    try:
        return ' '.join(ast.literal_eval(raw))
    except Exception:
        return str(raw)

def classificar_nota(score):
    if score <= 400: return 'Insuficiente'
    elif score <= 600: return 'Regular'
    elif score <= 800: return 'Bom'
    else: return 'Excelente'

print("Carregando dados...")
df_bert = pd.read_csv(CSV_BERT)
df_original = pd.read_csv(CSV_ORIGINAL)
df_original['score_class'] = df_original['score'].apply(classificar_nota)
df_original['texto'] = df_original['essay'].apply(parse_essay)
df_original['num_palavras'] = df_original['texto'].apply(lambda t: len(t.split()))

# Confirma que as duas bases estão na mesma ordem (mesmo número de linhas)
assert len(df_bert) == len(df_original), "As duas bases têm tamanhos diferentes — confira se são do mesmo dataset"

X = df_bert.drop(columns=['score_class']).values
y = df_bert['score_class']

print("Normalizando e aplicando PCA (95% variância)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Treinando SMO (o melhor modelo do projeto) com cross-validation...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
smo = SVC(C=1.0, kernel='linear', random_state=42)
y_pred = cross_val_predict(smo, X_pca, y, cv=cv)

# ── Monta dataframe de análise ──────────────────────────────────
analise = pd.DataFrame({
    'score_real_numerico': df_original['score'].values,
    'classe_real': y.values,
    'classe_prevista': y_pred,
    'num_palavras': df_original['num_palavras'].values,
    'acertou': (y.values == y_pred)
})

print(f"\nAcurácia geral confirmada: {analise['acertou'].mean()*100:.2f}%")

# ═══════════════════════════════════════════════════════════
# PERGUNTA 1 — O modelo erra mais em textos curtos ou longos?
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PERGUNTA 1 — Tamanho do texto vs Acerto")
print(f"{'='*70}")
print("\nMédia de palavras — Acertos vs Erros:")
print(analise.groupby('acertou')['num_palavras'].agg(['mean', 'median', 'std']).round(1))

# Divide em quartis de tamanho e vê a taxa de acerto em cada
analise['quartil_tamanho'] = pd.qcut(analise['num_palavras'], 4, labels=['Q1 (mais curtas)', 'Q2', 'Q3', 'Q4 (mais longas)'])
print("\nTaxa de acerto por quartil de tamanho:")
print(analise.groupby('quartil_tamanho')['acertou'].mean().round(4) * 100)

# ═══════════════════════════════════════════════════════════
# PERGUNTA 2 — Redações "fronteira" (perto do limite) erram mais?
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PERGUNTA 2 — Distância até o limite de classe vs Acerto")
print(f"{'='*70}")

limites = [400, 600, 800]
def distancia_limite(score):
    return min(abs(score - l) for l in limites)

analise['distancia_limite'] = analise['score_real_numerico'].apply(distancia_limite)
analise['proximo_limite'] = analise['distancia_limite'] <= 50  # dentro de 50 pontos de um limite

print("\nTaxa de acerto: redações PRÓXIMAS de um limite vs DISTANTES:")
print(analise.groupby('proximo_limite')['acertou'].agg(['mean', 'count']))

# ═══════════════════════════════════════════════════════════
# PERGUNTA 3 — Quais classes confundem mais com quais?
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PERGUNTA 3 — Para onde vão os erros de cada classe?")
print(f"{'='*70}")
erros = analise[~analise['acertou']]
print("\nQuando o modelo erra a classe REAL, o que ele prevê no lugar?")
print(pd.crosstab(erros['classe_real'], erros['classe_prevista']))

# ═══════════════════════════════════════════════════════════
# Salva o dataframe completo para inspeção manual de casos extremos
# ═══════════════════════════════════════════════════════════
analise_completa = analise.copy()
analise_completa['texto_preview'] = df_original['texto'].str[:200].values  # primeiros 200 caracteres
analise_completa.to_csv('analise_erros_detalhada.csv', index=False)
print("\n✅ Salvo: analise_erros_detalhada.csv (com preview do texto de cada redação)")

print("""
=============================================================
COMO USAR ISSO NA APRESENTAÇÃO:
=============================================================
- Se a Pergunta 1 mostrar que textos MUITO CURTOS têm taxa de
  acerto menor, é um achado relevante: o modelo precisa de
  "material" suficiente para capturar o estilo da redação.

- Se a Pergunta 2 confirmar que redações PRÓXIMAS do limite de
  classe erram muito mais, isso é esperado estatisticamente
  (são casos ambíguos mesmo para um avaliador humano) — vale
  mencionar que parte do "erro" do modelo é inerente à
  dificuldade da própria tarefa de classificação em faixas.

- A Pergunta 3 (matriz de para-onde-vão-os-erros) mostra se as
  confusões são "vizinhas" (Bom confundido com Regular, o que é
  mais aceitável) ou "distantes" (Insuficiente confundido com
  Excelente, o que seria mais grave e raro).

- Abra o analise_erros_detalhada.csv e procure 2-3 exemplos de
  redações que o modelo errou MUITO (ex: Insuficiente prevista
  como Excelente) para mostrar um caso concreto na apresentação
  — números genéricos ficam mais interessantes com 1 exemplo real.
=============================================================
""")
