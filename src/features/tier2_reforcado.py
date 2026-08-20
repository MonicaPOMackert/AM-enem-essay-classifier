"""
=============================================================
TIER 2 REFORÇADO — POS Tags Completas + Lematização + Bigramas
=============================================================
Baseado no material que o professor enviou. Esse script ADICIONA
features que faltavam no Tier 2 original.

O QUE JÁ TÍNHAMOS no Tier 2 original:
 - POS tagging de 9 classes gramaticais (NOUN, VERB, ADJ, ADV,
   PROPN, PRON, CCONJ, SCONJ, NUM)
 - Lematização + TF-IDF ampliado (1000 features, bigramas)

O QUE ESTE SCRIPT ADICIONA (pedido do professor):
 - Mais classes gramaticais: AUX (verbos auxiliares), DET
   (determinantes), INTJ (interjeições) — o professor pediu pra
   olhar a lista completa de ~15 colunas de POS, não só 9
 - Razões/índices adicionais: Substantivo/Total, Pronome/Total
   (indica formalidade — textos com muitos pronomes pessoais
   tendem a ser mais informais/narrativos)
 - Complexidade sintática: proporção de conjunções subordinativas
   (SCONJ) em relação a coordenativas (CCONJ) — mede o quão
   "argumentativa" é a estrutura das frases (dissertações usam
   mais subordinação)

Por que isso é importante pro professor:
 Esse Tier também é interpretável — cada métrica gramatical tem
 significado linguístico claro, diferente do BERT.

ANTES DE RODAR:
 pip install pandas numpy scikit-learn spacy
 python -m spacy download pt_core_news_sm

ATENÇÃO: esse script demora mais que o Tier 1 (processa cada
redação com SpaCy). Para 4.570 redações, espere ~5-10 minutos.
=============================================================
"""
import pandas as pd
import numpy as np
import ast
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# ── AJUSTE AQUI O CAMINHO DO SEU ARQUIVO ─────────────────────
CSV_PATH = 'essay-br(4570TEXTOS) (2).csv'   # <-- coloque o caminho real no seu PC

print("Carregando modelo SpaCy (pt_core_news_sm)...")
nlp = spacy.load('pt_core_news_sm')
nlp.max_length = 2_000_000

STOPWORDS_PT = set([
    'de','da','do','das','dos','em','na','no','nas','nos','ao','aos',
    'para','por','pela','pelo','pelas','pelos','com','sem','sob','sobre',
    'entre','ate','apos','ante','desde','perante','per','tras','a','as',
    'o','os','um','uma','uns','umas','e','ou','mas','se','que',
    'nao','nem','ja','ainda','muito','mais','menos','bem','mal',
    'todo','toda','todos','todas','este','esta','esse','essa','isso',
    'ele','ela','eles','elas','eu','voce','nos','me','te','se','lhe',
])

def parse_essay(raw):
    try:
        partes = ast.literal_eval(raw)
        return ' '.join(partes)
    except Exception:
        return str(raw)

def classificar_nota(score):
    if score <= 400: return 'Insuficiente'
    elif score <= 600: return 'Regular'
    elif score <= 800: return 'Bom'
    else: return 'Excelente'

# ═══════════════════════════════════════════════════════════
# POS Tags EXPANDIDAS (pedido do professor: lista mais completa)
# ═══════════════════════════════════════════════════════════
# Já tínhamos: NOUN, VERB, ADJ, ADV, PROPN, PRON, CCONJ, SCONJ, NUM
# NOVO: AUX, DET, INTJ, PUNCT (separado, não junto no texto bruto)
TAGS_ORIGINAIS = ['NOUN','VERB','ADJ','ADV','PROPN','PRON','CCONJ','SCONJ','NUM']
TAGS_NOVAS = ['AUX', 'DET', 'INTJ']
TAGS_TODAS = TAGS_ORIGINAIS + TAGS_NOVAS

def pos_features_reforcadas(doc):
    total = max(len(doc), 1)
    contagem = {tag: 0 for tag in TAGS_TODAS}
    for token in doc:
        if token.pos_ in contagem:
            contagem[token.pos_] += 1

    feats = {}
    for tag, cnt in contagem.items():
        feats[f'pos_{tag.lower()}_count'] = cnt
        feats[f'pos_{tag.lower()}_ratio'] = round(cnt / total, 4)

    noun = contagem['NOUN']
    verb = contagem['VERB']
    adj = contagem['ADJ']
    pron = contagem['PRON']
    sconj = contagem['SCONJ']
    cconj = contagem['CCONJ']

    # Já tínhamos
    feats['ratio_adj_noun'] = round(adj / max(noun, 1), 4)
    feats['ratio_verb_noun'] = round(verb / max(noun, 1), 4)

    # NOVO: proporção de pronomes (indica formalidade/informalidade)
    feats['ratio_pron_total'] = round(pron / total, 4)

    # NOVO: complexidade sintática — subordinação vs coordenação
    # Dissertações argumentativas bem estruturadas tendem a usar mais
    # subordinação (porque, embora, visto que) do que coordenação simples (e, mas)
    total_conjuncoes = sconj + cconj
    feats['ratio_subordinacao'] = round(sconj / max(total_conjuncoes, 1), 4)

    return feats

def lematizar(doc):
    lemas = [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct and not token.is_space
        and token.lemma_.lower() not in STOPWORDS_PT
        and len(token.lemma_) > 2
    ]
    return ' '.join(lemas)

# ═══════════════════════════════════════════════════════════
# Pipeline principal
# ═══════════════════════════════════════════════════════════
print("Carregando dados...")
df = pd.read_csv(CSV_PATH)
df['texto'] = df['essay'].apply(parse_essay)
df['score_class'] = df['score'].apply(classificar_nota)

print("Processando textos com SpaCy (pode demorar 5-10 min)...")
docs = list(nlp.pipe(df['texto'], batch_size=50))
print("SpaCy concluído!")

print("Extraindo POS features reforçadas (com AUX, DET, INTJ)...")
pos_df = pd.DataFrame([pos_features_reforcadas(doc) for doc in docs])

print("\nNovas colunas adicionadas ao Tier 2:")
novas_cols = [c for c in pos_df.columns if any(
    tag.lower() in c for tag in TAGS_NOVAS
) or c in ['ratio_pron_total', 'ratio_subordinacao']]
print(novas_cols)

print("Lematizando textos para TF-IDF...")
textos_lematizados = [lematizar(doc) for doc in docs]

print("Calculando TF-IDF sobre lemas (1000 features, uni+bigrams)...")
tfidf = TfidfVectorizer(
    max_features=1000, ngram_range=(1, 2), min_df=5, strip_accents='unicode'
)
tfidf_matrix = tfidf.fit_transform(textos_lematizados)
tfidf_cols = [f'tfidf_{w}' for w in tfidf.get_feature_names_out()]
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols)

# ═══════════════════════════════════════════════════════════
# Junta tudo
# ═══════════════════════════════════════════════════════════
final_df = pd.concat([pos_df, tfidf_df], axis=1)
final_df['score_class'] = df['score_class'].values

print(f"\nShape Tier 2 reforçado: {final_df.shape}")
final_df.to_csv('tier2_reforcado.csv', index=False)
print("✅ Salvo: tier2_reforcado.csv")

# ═══════════════════════════════════════════════════════════
# TESTE RÁPIDO — Random Forest pra já ver se as features novas ajudam
# ═══════════════════════════════════════════════════════════
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
kappa_scorer = make_scorer(cohen_kappa_score)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

X = final_df.drop(columns=['score_class']).values
y = final_df['score_class'].values

print("\n" + "="*65)
print("TESTE RÁPIDO — Random Forest 100 árvores (10-fold CV)")
print("="*65)
acc = cross_val_score(rf, X, y, cv=cv, scoring='accuracy').mean()
kappa = cross_val_score(rf, X, y, cv=cv, scoring=kappa_scorer).mean()
print(f"Tier 2 reforçado | Acurácia: {acc*100:.2f}%  | Kappa: {kappa:.4f}")

print("""
=============================================================
COMPARAR COM O RESULTADO ORIGINAL (já tínhamos no slide):
=============================================================
Tier 2 ORIGINAL (sem as features novas, RandomForest 100 árvores):
  Acurácia: 55,30%  |  Kappa: 0,270

Se o Tier 2 REFORÇADO superar isso, prova que detalhar mais a
análise gramatical (AUX, DET, INTJ, subordinação) agrega valor —
e continua sendo interpretável, diferente do BERT.

DICA PRA APRESENTAÇÃO: a razão de subordinação é um ótimo ponto
de discussão linguística — dissertações mais "argumentativas"
tendem a ter mais conjunções subordinativas (porque, visto que,
embora) que conjunções coordenativas simples (e, mas, ou).
=============================================================
""")
