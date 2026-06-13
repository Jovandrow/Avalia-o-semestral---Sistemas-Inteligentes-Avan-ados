# Questão 2 — Wine Quality (Classificação)
# Dataset: https://archive.ics.uci.edu/dataset/186/wine+quality

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ── 1. CARREGAMENTO E MONTAGEM DO DATASET ────────────────────────────────────
url_red   = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

red   = pd.read_csv(url_red,   sep=';')
white = pd.read_csv(url_white, sep=';')

red['type']   = 0   # tinto
white['type'] = 1   # branco

df = pd.concat([red, white], ignore_index=True)
print("Shape combinado:", df.shape)
print(df.head())

# ── 2. PRÉ-PROCESSAMENTO ─────────────────────────────────────────────────────
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

X = df.drop(columns=['quality'])
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── 3. PIPELINE + AVALIAÇÃO DE 3 MODELOS ─────────────────────────────────────
modelos = {
    'RandomForest'      : RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting'  : GradientBoostingClassifier(n_estimators=100, random_state=42),
    'KNN'               : KNeighborsClassifier(n_neighbors=7),
}

resultados = {}
for nome, clf in modelos.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    resultados[nome] = {'pipeline': pipe, 'acc': acc, 'f1': f1, 'pred': y_pred}
    print(f"\n{nome}: Acurácia={acc:.4f}  F1={f1:.4f}")

# ── 4. MELHOR MODELO ─────────────────────────────────────────────────────────
melhor_nome = max(resultados, key=lambda n: resultados[n]['f1'])
melhor      = resultados[melhor_nome]
print(f"\n✔ Melhor modelo: {melhor_nome} (F1={melhor['f1']:.4f})")
# Justificativa: maior F1-score ponderado indica melhor equilíbrio
# entre precisão e recall em todas as classes de qualidade.

# ── 5. MATRIZ DE CONFUSÃO DO MELHOR MODELO ───────────────────────────────────
cm = confusion_matrix(y_test, melhor['pred'])
disp = ConfusionMatrixDisplay(cm, display_labels=sorted(y.unique()))
disp.plot(cmap='Blues'); plt.title(f'Confusão — {melhor_nome}')
plt.tight_layout(); plt.savefig('confusao_q2.png'); plt.close()

# Acurácia por classe (via diagonal da matriz normalizada)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
for i, cls in enumerate(sorted(y.unique())):
    print(f"  Acurácia classe {cls}: {cm_norm[i, i]:.4f}")

# ── 6. INFERÊNCIA ─────────────────────────────────────────────────────────────
def inferir_qualidade(amostra: dict):
    row = pd.DataFrame([amostra])
    pred = melhor['pipeline'].predict(row)[0]
    print(f"Qualidade predita: {pred}")
    return pred

amostra_exemplo = {
    'fixed acidity': 7.4, 'volatile acidity': 0.7, 'citric acid': 0.0,
    'residual sugar': 1.9, 'chlorides': 0.076, 'free sulfur dioxide': 11.0,
    'total sulfur dioxide': 34.0, 'density': 0.9978, 'pH': 3.51,
    'sulphates': 0.56, 'alcohol': 9.4, 'type': 0
}
inferir_qualidade(amostra_exemplo)
