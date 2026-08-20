"""
=============================================================
TIER 3 — Embeddings Word2Vec (NILC/USP) + BERTimbau
=============================================================
ANTES DE RODAR:

  1) Word2Vec NILC (gratuito):
     Baixe o modelo em: http://nilc.icmc.usp.br/embeddings
     Recomendado: cbow_s100.zip  (100 dimensões, mais leve)
     Extraia e anote o caminho abaixo em WORD2VEC_PATH

  2) BERTimbau:
     pip install transformers torch sentencepiece
     (baixa automaticamente do HuggingFace na primeira execução)

Features geradas:
  - Word2Vec: média dos vetores de cada palavra (100 colunas)
  - BERTimbau: vetor [CLS] da última camada (768 colunas)

Saída: redacoes_tier3_w2v.arff e redacoes_tier3_bert.arff
=============================================================
"""

import pandas as pd
import numpy as np
import ast
import re

# ── Configuração ─────────────────────────────────────────────
WORD2VEC_PATH = 'cbow_s100.txt'   # <-- ajuste para o caminho do seu arquivo NILC
CSV_PATH      = 'essay-br_4570TEXTOS___1_.csv'  # <-- ajuste se necessário

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

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

def salvar_arff(df, nome_relacao, arff_path):
    classes  = ['Insuficiente','Regular','Bom','Excelente']
    num_cols = [c for c in df.columns if c != 'score_class']
    with open(arff_path, 'w', encoding='utf-8') as f:
        f.write(f"@RELATION {nome_relacao}\n\n")
        for col in num_cols:
            f.write(f"@ATTRIBUTE {col} NUMERIC\n")
        f.write(f"@ATTRIBUTE score_class {{{','.join(classes)}}}\n\n")
        f.write("@DATA\n")
        for _, row in df.iterrows():
            vals = [str(round(float(row[c]), 6)) for c in num_cols]
            vals.append(row['score_class'])
            f.write(','.join(vals) + '\n')
    print(f"ARFF salvo: {arff_path}")

# ── Carrega dados ────────────────────────────────────────────
print("Carregando dados...")
df = pd.read_csv(CSV_PATH)
df['texto']       = df['essay'].apply(parse_essay)
df['texto_limpo'] = df['texto'].apply(limpar_texto)
df['score_class'] = df['score'].apply(classificar_nota)

# ============================================================
# PARTE A — Word2Vec NILC
# ============================================================
print("\n=== PARTE A: Word2Vec NILC ===")
print(f"Carregando vetores de: {WORD2VEC_PATH}")
print("(pode demorar 1-2 minutos dependendo do arquivo...)")

word_vectors = {}
with open(WORD2VEC_PATH, 'r', encoding='utf-8') as f:
    primeira_linha = f.readline().strip().split()
    vocab_size, dim = int(primeira_linha[0]), int(primeira_linha[1])
    print(f"Vocabulário: {vocab_size} palavras, dimensão: {dim}")
    for linha in f:
        partes = linha.strip().split()
        if len(partes) < dim + 1:
            continue
        palavra = partes[0]
        vetor   = np.array(partes[1:dim+1], dtype=np.float32)
        word_vectors[palavra] = vetor

print(f"Vetores carregados: {len(word_vectors)}")

def texto_para_vetor_w2v(texto, dim):
    """Média dos vetores Word2Vec das palavras do texto."""
    palavras = texto.split()
    vetores  = [word_vectors[p] for p in palavras if p in word_vectors]
    if not vetores:
        return np.zeros(dim)
    return np.mean(vetores, axis=0)

print("Calculando embeddings Word2Vec para todas as redações...")
w2v_matrix = np.array([texto_para_vetor_w2v(t, dim) for t in df['texto_limpo']])

w2v_cols = [f'w2v_{i}' for i in range(dim)]
w2v_df   = pd.DataFrame(w2v_matrix, columns=w2v_cols)
w2v_df['score_class'] = df['score_class'].values

print(f"Shape Word2Vec: {w2v_df.shape}")
w2v_df.to_csv('redacoes_tier3_w2v.csv', index=False)
salvar_arff(w2v_df, 'redacoes_tier3_w2v', 'redacoes_tier3_w2v.arff')
print("Word2Vec CONCLUÍDO!")

# ============================================================
# PARTE B — BERTimbau
# ============================================================
print("\n=== PARTE B: BERTimbau ===")
print("Instalando/carregando BERTimbau (neuralmind/bert-base-portuguese-cased)...")
print("ATENÇÃO: primeira execução baixa ~500MB do HuggingFace")

try:
    import torch
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    model     = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Usando device: {device}")

    def texto_para_vetor_bert(texto, max_len=512):
        """Extrai vetor [CLS] do BERTimbau (representa o texto inteiro)."""
        inputs = tokenizer(
            texto,
            return_tensors='pt',
            max_length=max_len,
            truncation=True,
            padding='max_length'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Vetor do token [CLS] da última camada
        cls_vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_vector

    BATCH_SIZE = 16
    bert_vetores = []

    print(f"Processando {len(df)} redações em batches de {BATCH_SIZE}...")
    for i in range(0, len(df), BATCH_SIZE):
        batch = df['texto'].iloc[i:i+BATCH_SIZE].tolist()
        for texto in batch:
            bert_vetores.append(texto_para_vetor_bert(texto))
        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"  Batch {i // BATCH_SIZE + 1}/{len(df)//BATCH_SIZE + 1}")

    bert_matrix = np.array(bert_vetores)
    bert_cols   = [f'bert_{i}' for i in range(bert_matrix.shape[1])]
    bert_df     = pd.DataFrame(bert_matrix, columns=bert_cols)
    bert_df['score_class'] = df['score_class'].values

    print(f"Shape BERTimbau: {bert_df.shape}")
    bert_df.to_csv('redacoes_tier3_bert.csv', index=False)
    salvar_arff(bert_df, 'redacoes_tier3_bert', 'redacoes_tier3_bert.arff')
    print("BERTimbau CONCLUÍDO!")

except ImportError:
    print("ERRO: torch ou transformers não instalados.")
    print("Rode: pip install torch transformers sentencepiece")

print("\n=== TIER 3 CONCLUÍDO ===")
print("Arquivos gerados:")
print("  redacoes_tier3_w2v.arff  (Word2Vec NILC)")
print("  redacoes_tier3_bert.arff (BERTimbau)")
