# Questão 3 — Black Friday Sales (Multi-target: 3 classificadores)
# Dataset: https://www.kaggle.com/datasets/noopurbhatt/retail-black-friday-sales-dataset
# !! Baixe o CSV do Kaggle e ajuste o caminho em CSV_PATH !!

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

CSV_PATH = "black_friday.csv"   # ← ajuste o caminho

# ── 1. CARREGAMENTO ───────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(df.head()); print(df.dtypes)

# ── 2. PRÉ-PROCESSAMENTO ─────────────────────────────────────────────────────
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Codificar todas as colunas categóricas com LabelEncoder
le_map = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_map[col] = le

# Alvos
TARGETS = ['product_category', 'payment_method', 'age_group']

# Features: todas as colunas que não são alvo
FEATURES = [c for c in df.columns if c not in TARGETS]

# ── 3. FUNÇÃO GENÉRICA DE TREINO + MÉTRICAS ───────────────────────────────────
def treinar(alvo: str):
    X = df[FEATURES]
    y = df[alvo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Especificidade e Sensibilidade por classe via CM
    cm = confusion_matrix(y_test, y_pred)
    sens_list, spec_list = [], []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens_list.append(tp / (tp + fn) if (tp + fn) else 0)
        spec_list.append(tn / (tn + fp) if (tn + fp) else 0)

    print(f"\n── {alvo.upper()} ──")
    print(f"  Acurácia Global : {acc:.4f}")
    print(f"  Sensibilidade   : {np.mean(sens_list):.4f}")
    print(f"  Especificidade  : {np.mean(spec_list):.4f}")
    print(f"  F1-Score        : {f1:.4f}")

    # Matriz de confusão
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap='Blues'); plt.title(f'Confusão — {alvo}')
    plt.tight_layout(); plt.savefig(f'confusao_{alvo}.png'); plt.close()

    return pipe

# ── 4. TREINO DOS 3 MODELOS ───────────────────────────────────────────────────
pipes = {alvo: treinar(alvo) for alvo in TARGETS}

# ── 5. INFERÊNCIA COM GRAU DE CERTEZA ────────────────────────────────────────
def inferir(venda: dict):
    """
    venda: dict com as colunas de FEATURES.
    Retorna predição + probabilidade para cada alvo.
    """
    row = pd.DataFrame([venda])
    print("\n── Resultado da Inferência ──")
    for alvo, pipe in pipes.items():
        pred  = pipe.predict(row)[0]
        proba = pipe.predict_proba(row).max()
        # Decodificar label se existir encoder
        label = le_map[alvo].inverse_transform([pred])[0] if alvo in le_map else pred
        print(f"  {alvo:20s}: {label}  (certeza: {proba:.1%})")

# Exemplo — preencha com valores reais do seu dataset
venda_exemplo = {col: df[col].iloc[0] for col in FEATURES}
inferir(venda_exemplo)
