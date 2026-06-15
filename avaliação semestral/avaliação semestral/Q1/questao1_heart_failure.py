# Questão 1 — Heart Failure (Agrupamento com KMeans)
# Dataset: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ── 1. CARREGAMENTO ──────────────────────────────────────────────────────────
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
print(df.head())
print(df.dtypes)

# ── 2. PRÉ-PROCESSAMENTO ─────────────────────────────────────────────────────
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

binary_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']
numeric_cols = [c for c in df.columns if c not in binary_cols]

print("Colunas numéricas contínuas:", numeric_cols)
print("Colunas binárias (mantidas como 0/1):", binary_cols)


# ── 3. PIPELINE ──────────────────────────────────────────────────────────────
# Só as numéricas são escaladas; binárias entram direto 
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(df[numeric_cols])
X_final = np.hstack([X_num_scaled, df[binary_cols].values])

# Método do cotovelo
inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_final).inertia_
            for k in range(2, 9)]
plt.plot(range(2, 9), inertias, marker='o')
plt.title('Método do Cotovelo'); plt.xlabel('k'); plt.ylabel('Inércia')
plt.tight_layout(); plt.savefig('cotovelo_q1.png'); plt.close()

# k=3 escolhido pelo cotovelo
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_final)
print("\nDistribuição dos clusters:\n", df['cluster'].value_counts())
print("\nMédia por cluster:\n", df.groupby('cluster').mean().round(2))

# ── 4. VISUALIZAÇÃO PCA ───────────────────────────────────────────────────────
pca = PCA(n_components=2)
coords = pca.fit_transform(X_final)
plt.figure(figsize=(7, 5))
for c in range(k):
    mask = df['cluster'] == c
    plt.scatter(coords[mask, 0], coords[mask, 1], label=f'Cluster {c}', s=20)
plt.title('Clusters (PCA 2D)'); plt.legend(); plt.tight_layout()
plt.savefig('clusters_q1.png'); plt.close()

# ── 5. INFERÊNCIA ─────────────────────────────────────────────────────────────
def inferir_cluster(paciente: dict):
    """
    paciente: dict com as mesmas colunas do dataset (sem 'cluster').
    Retorna o cluster ao qual o paciente pertence.
    """
    row = pd.DataFrame([paciente])
    row_num_scaled = scaler.transform(row[numeric_cols])
    row_bin = row[binary_cols].values
    row_final = np.hstack([row_num_scaled, row_bin])
    cluster = kmeans.predict(row_final)[0]
    print(f"Paciente pertence ao Cluster {cluster}")
    return cluster

# Exemplo
paciente_exemplo = {
    'age': 65, 'creatinine_phosphokinase': 582, 'ejection_fraction': 20,
    'platelets': 265000, 'serum_creatinine': 1.9, 'serum_sodium': 130,
    'time': 4,
    'anaemia': 0, 'diabetes': 0, 'high_blood_pressure': 1,
    'sex': 1, 'smoking': 0, 'DEATH_EVENT': 0
}
inferir_cluster(paciente_exemplo)
