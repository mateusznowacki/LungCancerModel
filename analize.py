import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# === Katalog wyjściowy ===
output_dir = "analysis_plots"
os.makedirs(output_dir, exist_ok=True)

# === Wczytywanie plików CSV ===
features_df = pd.read_csv("mammography.csv", header=None)
structured_df = pd.read_csv("dataset_prepared.csv")

# === Podział na cechy i etykiety ===
X1 = features_df.iloc[:, :-1]
y1 = features_df.iloc[:, -1]

X2 = structured_df.drop(columns=["Patient Id", "Level", "Result"])
y2 = structured_df["Result"]

# === Łączenie danych ===
X = pd.concat([X1, pd.DataFrame(X2)], ignore_index=True)
y = pd.concat([pd.Series(y1), pd.Series(y2)], ignore_index=True)

# Nazwy kolumn jako stringi (dla sklearn)
X.columns = [f"Feature_{i}" for i in range(X.shape[1])]

# Usunięcie brakujących danych
X = X.dropna()
y = y[X.index]  # Dopasuj etykiety do przefiltrowanych danych

# === Statystyki opisowe ===
X.describe().to_csv(os.path.join(output_dir, "statystyki_opisowe.csv"))
print("✔ Statystyki zapisane")

# === Histogramy ===
X.hist(bins=20, figsize=(15, 10))
plt.suptitle("Histogramy cech")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "histogramy_cech.png"))
plt.close()

# === Heatmapa korelacji ===
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
plt.title("Macierz korelacji cech")
plt.savefig(os.path.join(output_dir, "macierz_korelacji.png"))
plt.close()

# === PCA ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Result"] = y.values
pca_df.to_csv(os.path.join(output_dir, "pca_dane.csv"), index=False)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Result", palette="bwr", alpha=0.7)
plt.title("PCA - Wizualizacja klas (Result)")
plt.savefig(os.path.join(output_dir, "pca_wykres.png"))
plt.close()

# === Boxploty (cechy vs klasy) ===
X_box = X.copy()
X_box["Result"] = y.values
for col in X.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=X_box, x="Result", y=col)
    plt.title(f"Boxplot: {col} vs Result")
    plt.savefig(os.path.join(output_dir, f"boxplot_{col}.png"))
    plt.close()

# === KDE - gęstość rozkładów dla 3 pierwszych cech ===
for col in X.columns[:3]:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=X_box, x=col, hue="Result", fill=True, common_norm=False, alpha=0.5)
    plt.title(f"KDE: {col} vs Result")
    plt.savefig(os.path.join(output_dir, f"kde_{col}.png"))
    plt.close()

print(f"📂 Analizy i wykresy zapisane w katalogu: {output_dir}/")
