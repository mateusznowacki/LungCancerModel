# lung_cancer_training.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import os

# === Wczytanie danych ===
df = pd.read_csv("dataset_smote.csv")

column_renames = {
    "Alcohol use": "Alcohol_Use",
    "Dust Allergy": "Dust_Allergy",
    "OccuPational Hazards": "Occupational_Hazards",
    "Genetic Risk": "Genetic_Risk",
    "chronic Lung Disease": "Chronic_Lung_Disease",
    "Balanced Diet": "Balanced_Diet",
    "Passive Smoker": "Passive_Smoker",
    "Chest Pain": "Chest_Pain",
    "Coughing of Blood": "Coughing_Blood",
    "Weight Loss": "Weight_Loss",
    "Shortness of Breath": "Shortness_Breath",
    "Swallowing Difficulty": "Swallowing_Difficulty",
    "Clubbing of Finger Nails": "Clubbing_Nails",
    "Frequent Cold": "Frequent_Cold",
    "Dry Cough": "Dry_Cough"
}
df = df.rename(columns=column_renames)
columns_to_drop = ["Patient Id", "Level", "Smoking.1", "Swallowing_Difficulty.1"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

print("Dostępne kolumny:", df.columns.tolist())
print("Unikalne klasy w Result:", df['Result'].unique())

X = df.drop(columns=["Result"]).values
y = df["Result"].values

encoder = LabelEncoder()
y = encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Model 1: XGBoost ===
xgb_model = xgb.XGBClassifier(objective="binary:logistic", max_depth=6, learning_rate=0.1, n_estimators=200)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds)
xgb_prec = precision_score(y_test, xgb_preds)
xgb_rec = recall_score(y_test, xgb_preds)

print("[XGBoost] Evaluation:")
print(f"Accuracy: {xgb_acc:.4f} | F1: {xgb_f1:.4f} | Precision: {xgb_prec:.4f} | Recall: {xgb_rec:.4f}")
print(classification_report(y_test, xgb_preds, digits=4))

# === Model 2: Cost-Sensitive NN z trzema mechanizmami ===
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = torch.tensor([total_samples / (2.0 * c) for c in class_counts], dtype=torch.float32)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
test_ds = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

class CostSensitiveNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.out(x)

class CombinedCostSensitiveLoss(nn.Module):
    def __init__(self, weights, reg_weight=0.7, boost_weight=2.0):
        super().__init__()
        self.weights = weights
        self.reg_weight = reg_weight
        self.boost_weight = boost_weight

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        base_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weights = targets * self.weights[1] + (1 - targets) * self.weights[0]
        weighted_loss = base_loss * weights
        margin_penalty = targets * torch.clamp(1 - probs, min=0)
        boost_factor = 1 + self.boost_weight * torch.abs(targets - probs)
        total_loss = (weighted_loss + self.reg_weight * margin_penalty) * boost_factor
        return total_loss.mean()

model = CostSensitiveNet(X.shape[1])
loss_fn = CombinedCostSensitiveLoss(class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

best_f1 = 0
model_path = "model_best.pt"

for epoch in range(75):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = loss_fn(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)

    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = torch.sigmoid(model(xb))
            predicted = (preds > 0.5).int()
            preds_all.extend(predicted.cpu().numpy())
            labels_all.extend(yb.cpu().numpy())

    f1 = f1_score(labels_all, preds_all)
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), model_path)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f} | F1: {f1:.4f}")

# === Ocena najlepszego modelu ===
model.load_state_dict(torch.load(model_path))
model.eval()
preds_all, labels_all = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = torch.sigmoid(model(xb))
        predicted = (preds > 0.5).int()
        preds_all.extend(predicted.cpu().numpy())
        labels_all.extend(yb.cpu().numpy())

acc = accuracy_score(labels_all, preds_all)
f1 = f1_score(labels_all, preds_all)
prec = precision_score(labels_all, preds_all)
rec = recall_score(labels_all, preds_all)

print("[Cost-sensitive] Evaluation:")
print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
print("\n[Classification Report]\n")
print(classification_report(labels_all, preds_all, digits=4))

# === Wykres porównania ===
labels = ['Accuracy', 'F1-score', 'Precision', 'Recall']
xgb_scores = [xgb_acc, xgb_f1, xgb_prec, xgb_rec]
torch_scores = [acc, f1, prec, rec]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, xgb_scores, width, label='XGBoost')
plt.bar(x + width/2, torch_scores, width, label='Cost-sensitive (NN)')
plt.ylabel('Score')
plt.title('Porównanie wyników modeli')
plt.xticks(x, labels)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# === Predykcja na własnych danych ===
print("\nMożesz teraz wpisać własne dane do predykcji (0-7, liczby całkowite/z przecinkiem, wg kolejności kolumn):")
columns = df.drop(columns=["Result"]).columns.tolist()
user_input = []
for col in columns:
    val = float(input(f"{col}: "))
    user_input.append(val)

input_array = np.array(user_input).reshape(1, -1)
input_scaled = scaler.transform(input_array)
tensor_input = torch.tensor(input_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    output = torch.sigmoid(model(tensor_input))
    prediction = (output > 0.5).int().item()

print("\nWynik predykcji:")
if prediction == 1:
    print("✅ Wysokie prawdopodobieństwo raka płuc")
else:
    print("🟢 Niskie prawdopodobieństwo raka płuc")
