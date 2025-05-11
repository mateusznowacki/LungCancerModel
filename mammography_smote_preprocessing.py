import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# === Wczytaj dane ===
df = pd.read_csv("mammography.csv", header=None)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values

# Zamień -1/1 na 0/1
y = LabelEncoder().fit_transform(y)  # -1 → 0, 1 → 1

# Skalowanie
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Podział stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print("Przed SMOTE:", Counter(y_train))

# === SMOTE na zbiorze treningowym ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Po SMOTE:", Counter(y_train_resampled))

# === Zapisz nowe dane (opcjonalnie) ===
# Możesz zapisać do pliku CSV:
X_res_df = pd.DataFrame(X_train_resampled)
y_res_df = pd.DataFrame(y_train_resampled, columns=["target"])
df_smote = pd.concat([X_res_df, y_res_df], axis=1)
df_smote.to_csv("mammography_smote.csv", index=False)

print("\nZbiór `mammography_smote.csv` gotowy do treningu.")
