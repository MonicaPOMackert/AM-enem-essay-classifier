"""
=============================================================
TIER 1 — Meta-features de Superfície + TF-IDF
=============================================================
Features geradas:
  - Contagem de caracteres, palavras, tamanho médio de palavras
  - Proporção de maiúsculas e pontuação
  - TF-IDF com max 500 features (stopwords PT removidas)

Saída: redacoes_tier1.arff  (pronto para o Weka)
=============================================================
"""

import pandas as pd
import numpy as np
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Stopwords PT (sem depender de internet) ─────────────────
STOPWORDS_PT = set([
    'de','da','do','das','dos','em','na','no','nas','nos','ao','aos',
    'para','por','pela','pelo','pelas','pelos','com','sem','sob','sobre',
    'entre','ate','apos','ante','desde','perante','per','tras','a','as',
    'o','os','um','uma','uns','umas','e','eh','é','ou','mas','se','que',
    'nao','nem','ja','ainda','so','ate','muito','mais','menos','bem','mal',
    'todo','toda','todos','todas','este','esta','estes','estas','esse',
    'essa','esses','essas','aquele','aquela','aqueles','aquelas','isto',
    'isso','aquilo','ele','ela','eles','elas','eu','tu','voce','nos','vos',
    'me','te','se','lhe','nos','vos','lhes','meu','minha','meus','minhas',
    'teu','tua','teus','tuas','seu','sua','seus','suas','nosso','nossa',
    'como','quando','onde','porque','pois','logo','assim','entao',
    'num','numa','nele','nela','neles','nelas','dele','dela','deles','delas',
    'deste','desta','desse','dessa','disso','disto','daquele','daquela',
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

# ── Features de superfície ───────────────────────────────────

def meta_features(texto):
    palavras = texto.split()
    num_palavras  = len(palavras)
    num_chars     = len(texto)
    media_palavra = np.mean([len(p) for p in palavras]) if palavras else 0

    # Maiúsculas (exceto primeira letra de frase)
    letras = [c for c in texto if c.isalpha()]
    ratio_maiusculas = sum(1 for c in letras if c.isupper()) / len(letras) if letras else 0

    # Pontuação
    exclamacoes   = texto.count('!')
    interrogacoes = texto.count('?')
    virgulas      = texto.count(',')
    pontos        = texto.count('.')

    return {
        'num_palavras':        num_palavras,
        'num_caracteres':      num_chars,
        'media_chars_palavra': round(media_palavra, 4),
        'ratio_maiusculas':    round(ratio_maiusculas, 4),
        'num_exclamacoes':     exclamacoes,
        'num_interrogacoes':   interrogacoes,
        'num_virgulas':        virgulas,
        'num_pontos':          pontos,
    }

# ── Pipeline principal ───────────────────────────────────────

# ── AJUSTE AQUI O CAMINHO DO SEU ARQUIVO ─────────────────────
CSV_PATH = 'essay-br(4570TEXTOS) (2).csv'   # <-- coloque o caminho real no seu PC

print("Carregando dados...")
df = pd.read_csv(CSV_PATH)
df['texto'] = df['essay'].apply(parse_essay)
df['score_class'] = df['score'].apply(classificar_nota)

# Meta-features
print("Extraindo meta-features...")
meta_df = pd.DataFrame([meta_features(t) for t in df['texto']])

# TF-IDF (500 features, stopwords PT)
print("Calculando TF-IDF (500 features)...")
tfidf = TfidfVectorizer(
    max_features=500,
    stop_words=list(STOPWORDS_PT),
    strip_accents='unicode',
    lowercase=True,
    ngram_range=(1, 1)
)
tfidf_matrix = tfidf.fit_transform(df['texto'])
tfidf_cols   = [f'tfidf_{w}' for w in tfidf.get_feature_names_out()]
tfidf_df     = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols)

# Junta tudo
final_df = pd.concat([meta_df, tfidf_df], axis=1)
final_df['score_class'] = df['score_class'].values

print(f"Shape final: {final_df.shape}")
print("Distribuição:\n", final_df['score_class'].value_counts())

# ── Salva CSV ────────────────────────────────────────────────
csv_out = 'redacoes_tier1.csv'
final_df.to_csv(csv_out, index=False)
print(f"CSV salvo: {csv_out}")

# ── Salva ARFF ───────────────────────────────────────────────
arff_out = 'redacoes_tier1.arff'
classes  = ['Insuficiente','Regular','Bom','Excelente']
num_cols = [c for c in final_df.columns if c != 'score_class']

print("Gerando ARFF...")
with open(arff_out, 'w', encoding='utf-8') as f:
    f.write("@RELATION redacoes_tier1\n\n")
    for col in num_cols:
        f.write(f"@ATTRIBUTE {col} NUMERIC\n")
    f.write(f"@ATTRIBUTE score_class {{{','.join(classes)}}}\n\n")
    f.write("@DATA\n")
    for _, row in final_df.iterrows():
        vals = [str(round(row[c], 6)) for c in num_cols]
        vals.append(row['score_class'])
        f.write(','.join(vals) + '\n')

print(f"ARFF salvo: {arff_out}")
print("TIER 1 CONCLUÍDO!")
