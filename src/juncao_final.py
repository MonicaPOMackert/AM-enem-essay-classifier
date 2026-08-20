"""
=============================================================
JUNÇÃO FINAL v3 — Métricas completas + MCC (robustez p/ desbalanceamento)
=============================================================
EVOLUÇÃO da v2: adicionamos o MCC (Matthews Correlation
Coefficient) por classe — a métrica mais recomendada na
literatura para bases desbalanceadas, porque considera
TODAS as células da matriz de confusão (TP, TN, FP, FN) ao
mesmo tempo, diferente da Acurácia (que pode mentir quando
uma classe domina a base).

MÉTRICAS QUE ESSE SCRIPT CALCULA, por classe e em média:
 - Recall (TP Rate)      -> dos casos reais da classe, quantos achei
 - Precision             -> das vezes que apostei na classe, quantos acertei
 - F1-Score               -> equilíbrio entre Precision e Recall
 - MCC                    -> correlação entre predição e realidade
                             (-1 a +1, 0 = aleatório, 1 = perfeito)
 - Matriz de Confusão     -> visão completa de onde o modelo erra

Por que isso importa AQUI especificamente: nossa base é
desbalanceada (Regular+Bom = 78% da base, Insuficiente+Excelente
juntos são só 22%). Um modelo pode ter Acurácia "boa" só por
acertar sempre as classes majoritárias, escondendo que ele é
praticamente inútil pra identificar Insuficiente/Excelente.
O MCC penaliza isso, a Acurácia sozinha não.

PRÉ-REQUISITO: mesmos arquivos das versões anteriores, na mesma pasta:
 - tier1_reforcado_500.csv
 - tier2_reforcado.csv
 - feature_tema.csv

ANTES DE RODAR:
 pip install pandas numpy scikit-learn
=============================================================
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    precision_recall_fscore_support, matthews_corrcoef
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

ORDEM_CLASSES = ['Insuficiente', 'Regular', 'Bom', 'Excelente']

def mcc_por_classe(y_true, y_pred, classes):
    """
    MCC não tem versão nativa 'por classe' no sklearn (ele já é
    multiclasse por padrão). Para ter um MCC por classe (no estilo
    do Weka, que mostra MCC linha a linha), fazemos a abordagem
    'um-contra-todos': para cada classe, vira um problema binário
    (é dessa classe vs não é) e calculamos o MCC desse binário.
    """
    resultado = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for classe in classes:
        y_true_bin = (y_true == classe).astype(int)
        y_pred_bin = (y_pred == classe).astype(int)
        resultado[classe] = matthews_corrcoef(y_true_bin, y_pred_bin)
    return resultado

def avaliar_completo(nome, X, y):
    print(f"\n{'='*75}")
    print(f"RESULTADO COMPLETO — {nome}")
    print(f"{'='*75}")

    y_pred = cross_val_predict(rf, X, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    mcc_geral = matthews_corrcoef(y, y_pred)

    print(f"Acurácia: {acc*100:.2f}%  |  Kappa: {kappa:.4f}  |  MCC geral: {mcc_geral:.4f}")

    # ── Precision, Recall, F1 por classe ────────────────────────
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        y, y_pred, labels=ORDEM_CLASSES, zero_division=0
    )
    mcc_classes = mcc_por_classe(y, y_pred, ORDEM_CLASSES)

    print("\n--- Detailed Accuracy By Class (igual ao Weka, + MCC) ---")
    tabela_classes = pd.DataFrame({
        'Classe': ORDEM_CLASSES,
        'Recall (TP Rate)': np.round(recall, 3),
        'Precision': np.round(precisao, 3),
        'F1-Score': np.round(f1, 3),
        'MCC': [round(mcc_classes[c], 3) for c in ORDEM_CLASSES],
        'Suporte (n)': suporte
    })
    print(tabela_classes.to_string(index=False))

    precisao_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y, y_pred, labels=ORDEM_CLASSES, average='weighted', zero_division=0
    )
    print(f"\nWeighted Avg.  |  Precision: {precisao_w:.3f}  |  Recall: {recall_w:.3f}  "
          f"|  F1-Score: {f1_w:.3f}  |  MCC geral: {mcc_geral:.3f}")

    # ── Matriz de Confusão ───────────────────────────────────────
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y, y_pred, labels=ORDEM_CLASSES)
    cm_df = pd.DataFrame(cm, index=[f"real_{c}" for c in ORDEM_CLASSES],
                          columns=[f"prev_{c}" for c in ORDEM_CLASSES])
    print(cm_df.to_string())

    return {
        'experimento': nome,
        'acuracia': round(acc*100, 2),
        'kappa': round(kappa, 4),
        'mcc_geral': round(mcc_geral, 4),
        'precision_weighted': round(precisao_w, 4),
        'recall_weighted': round(recall_w, 4),
        'f1_weighted': round(f1_w, 4),
        'mcc_insuficiente': round(mcc_classes['Insuficiente'], 4),
        'mcc_excelente': round(mcc_classes['Excelente'], 4),
    }

resultados = []

print("="*75)
print("Carregando arquivos dos experimentos...")
print("="*75)

try:
    tier1 = pd.read_csv('tier1_reforcado_500.csv')
    print(f"✅ tier1_reforcado_500.csv carregado: {tier1.shape}")
except FileNotFoundError:
    tier1 = None
    print("⚠️  tier1_reforcado_500.csv não encontrado")

try:
    tier2 = pd.read_csv('tier2_reforcado.csv')
    print(f"✅ tier2_reforcado.csv carregado: {tier2.shape}")
except FileNotFoundError:
    tier2 = None
    print("⚠️  tier2_reforcado.csv não encontrado")

try:
    tema = pd.read_csv('feature_tema.csv')
    print(f"✅ feature_tema.csv carregado: {tema.shape}")
except FileNotFoundError:
    tema = None
    print("⚠️  feature_tema.csv não encontrado")

if tier1 is not None:
    X = tier1.drop(columns=['score_class']).values
    y = tier1['score_class'].values
    resultados.append(avaliar_completo("Tier 1 reforçado (sozinho)", X, y))

if tier1 is not None and tema is not None:
    tier1_com_tema = pd.concat(
        [tier1.drop(columns=['score_class']), tema[['feature_aderencia_tema']]], axis=1
    )
    X = tier1_com_tema.values
    y = tier1['score_class'].values
    resultados.append(avaliar_completo("Tier 1 reforçado + feature de tema", X, y))

if tier2 is not None:
    X = tier2.drop(columns=['score_class']).values
    y = tier2['score_class'].values
    resultados.append(avaliar_completo("Tier 2 reforçado (sozinho)", X, y))

if tier2 is not None and tema is not None:
    tier2_com_tema = pd.concat(
        [tier2.drop(columns=['score_class']), tema[['feature_aderencia_tema']]], axis=1
    )
    X = tier2_com_tema.values
    y = tier2['score_class'].values
    resultados.append(avaliar_completo("Tier 2 reforçado + feature de tema", X, y))

print(f"\n{'='*75}")
print("TABELA COMPARATIVA FINAL (RESUMO)")
print(f"{'='*75}")
resumo = pd.DataFrame(resultados)
print(resumo.to_string(index=False))
resumo.to_csv('resultado_final_completo_v3.csv', index=False)
print("\n✅ Salvo: resultado_final_completo_v3.csv")

print("""
=============================================================
POR QUE TRAZER ESSAS MÉTRICAS PRA APRESENTAÇÃO (não só Acurácia):
=============================================================
Como vimos no curso, Acurácia pode enganar em bases desbalanceadas.
Exemplo real do nosso projeto: um modelo "bobo" que sempre chuta
"Regular" (a classe mais comum) já acerta ~40% das vezes (ZeroR)
SEM aprender nada. Isso mostra que olhar só pra Acurácia não conta
a história toda.

- RECALL por classe: revela se o modelo "esquece" classes raras
  (no nosso caso, Insuficiente e Excelente sempre tiveram Recall
  muito mais baixo que Regular/Bom).

- PRECISION por classe: revela se, quando o modelo aposta numa
  classe rara, ele costuma estar certo (geralmente SIM no nosso
  caso — Precision de Excelente é alta, mas Recall é baixo, ou
  seja: quando o modelo arrisca dizer "Excelente" ele quase
  sempre acerta, mas ele RARAMENTE arrisca).

- MCC: é a métrica mais "honesta" porque usa as 4 células da
  matriz de confusão (acertos e erros de TODAS as classes) numa
  única fórmula. MCC próximo de 0 = modelo não é melhor que o
  acaso pra aquela classe. MCC alto = correlação real entre
  previsão e realidade.

- F1-Score: bom resumo quando você quer 1 número só, mas que
  já equilibra Precision e Recall (mais informativo que só
  Acurácia).

RECOMENDAÇÃO PRA APRESENTAÇÃO FINAL: ao invés de só citar
"Acurácia de X%", mostrem a tabela completa por classe pelo
menos no(s) melhor(es) experimento(s) — isso demonstra que o
grupo entende as limitações da Acurácia, exatamente o tipo de
analise crítica que enriquece a nota de "qualidade do trabalho".
=============================================================
""")
