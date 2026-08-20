"""
=============================================================
TIER 2 — POS Tags + TF-IDF com Lematização (SpaCy)
=============================================================
ANTES DE RODAR, instale:
  pip install spacy scikit-learn pandas numpy
  python -m spacy download pt_core_news_sm

Features geradas:
  - Frequência de classes gramaticais (NOUN, VERB, ADJ, ADV...)
  - Ratios de POS (adjetivos/total, verbos/total, etc.)
  - TF-IDF sobre lemas (palavras na forma primitiva)

Saída: redacoes_tier2.arff  (pronto para o Weka)
=============================================================
"""

import pandas as pd
import numpy as np
import ast
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Carrega modelo SpaCy PT ──────────────────────────────────
print("Carregando modelo SpaCy (pt_core_news_sm)...")
nlp = spacy.load('pt_core_news_sm')
# Para textos longos:
nlp.max_length = 2_000_000

# ── Stopwords PT ─────────────────────────────────────────────
STOPWORDS_PT = set([
    'de','da','do','das','dos','em','na','no','nas','nos','ao','aos',
    'para','por','pela','pelo','pelas','pelos','com','sem','sob','sobre',
    'entre','ate','apos','ante','desde','perante','per','tras','a','as',
    'o','os','um','uma','uns','umas','e','ou','mas','se','que',
    'nao','nem','ja','ainda','muito','mais','menos','bem','mal',
    'todo','toda','todos','todas','este','esta','esse','essa','isso',
    'ele','ela','eles','elas','eu','voce','nos','me','te','se','lhe',
    'num','numa','nele','nela','dele','dela','deste','desta','desse','dessa',
    'neste','nesta','nesse','nessa','nisso','nisto','naquele','naquela'
])

# ── Utilitários ──────────────────────────────────────────────

def parse_essay(raw):
    try:
        partes = ast.literal_eval(raw)
        return ' '.join(partes)
    except:
        return str(raw)

def classificar_nota(score):
    if score <= 400:   return 'Insuficiente'
    elif score <= 600: return 'Regular'
    elif score <= 800: return 'Bom'
    else:              return 'Excelente'

# ── Features de POS ─────────────────────────────────────────

TAGS_INTERESSE = ['NOUN','VERB','ADJ','ADV','PROPN','PRON','CCONJ','SCONJ','NUM']

def pos_features(doc):
    total = max(len(doc), 1)
    contagem = {tag: 0 for tag in TAGS_INTERESSE}
    for token in doc:
        if token.pos_ in contagem:
            contagem[token.pos_] += 1

    feats = {}
    for tag, cnt in contagem.items():
        feats[f'pos_{tag.lower()}_count'] = cnt
        feats[f'pos_{tag.lower()}_ratio'] = round(cnt / total, 4)

    # Ratios derivados relevantes para redação
    adj  = contagem['ADJ']
    noun = contagem['NOUN']
    verb = contagem['VERB']
    feats['ratio_adj_noun']  = round(adj  / max(noun, 1), 4)  # riqueza descritiva
    feats['ratio_verb_noun'] = round(verb / max(noun, 1), 4)  # dinamismo

    return feats

def lematizar(doc):
    """Retorna texto com lemas, sem stopwords e sem pontuação."""
    lemas = [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct
        and not token.is_space
        and token.lemma_.lower() not in STOPWORDS_PT
        and len(token.lemma_) > 2
    ]
    return ' '.join(lemas)

# ── Pipeline principal ───────────────────────────────────────

# ── AJUSTE AQUI O CAMINHO DO SEU ARQUIVO ─────────────────────
CSV_PATH = 'essay-br(4570TEXTOS) (2).csv'   # <-- coloque o caminho real no seu PC

print("Carregando dados...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['essay']).reset_index(drop=True)
df['texto'] = df['essay'].apply(parse_essay)
df['score_class'] = df['score'].apply(classificar_nota)

# Processa com SpaCy em lotes (mais rápido)
print("Processando textos com SpaCy (pode demorar ~5-10 min)...")
docs = list(nlp.pipe(df['texto'], batch_size=50))
print("SpaCy concluído!")

# POS features
print("Extraindo POS features...")
pos_df = pd.DataFrame([pos_features(doc) for doc in docs])

# Lematização para TF-IDF
print("Lematizando textos para TF-IDF...")
textos_lematizados = [lematizar(doc) for doc in docs]

print("Calculando TF-IDF sobre lemas (1000 features, uni+bigrams)...")
tfidf = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=5,
    strip_accents='unicode'
)
tfidf_matrix = tfidf.fit_transform(textos_lematizados)
tfidf_cols   = [f'tfidf_{w}' for w in tfidf.get_feature_names_out()]
tfidf_df     = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols)

# Junta tudo
final_df = pd.concat([pos_df, tfidf_df], axis=1)
final_df['score_class'] = df['score_class'].values

print(f"Shape final: {final_df.shape}")
print("Distribuição:\n", final_df['score_class'].value_counts())

# ── Salva CSV ────────────────────────────────────────────────
final_df.to_csv('redacoes_tier2.csv', index=False)
print("CSV salvo: redacoes_tier2.csv")

# ── Salva ARFF ───────────────────────────────────────────────
classes  = ['Insuficiente','Regular','Bom','Excelente']
num_cols = [c for c in final_df.columns if c != 'score_class']

print("Gerando ARFF...")
with open('redacoes_tier2.arff', 'w', encoding='utf-8') as f:
    f.write("@RELATION redacoes_tier2\n\n")
    for col in num_cols:
        safe_col = col.replace(' ', '_').replace('(', '').replace(')', '')
        f.write(f"@ATTRIBUTE {safe_col} NUMERIC\n")
    f.write(f"@ATTRIBUTE score_class {{{','.join(classes)}}}\n\n")
    f.write("@DATA\n")
    for _, row in final_df.iterrows():
        vals = [str(round(float(row[c]), 6)) for c in num_cols]
        vals.append(row['score_class'])
        f.write(','.join(vals) + '\n')

print("ARFF salvo: redacoes_tier2.arff")
print("TIER 2 CONCLUÍDO!")
