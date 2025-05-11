import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt

# === Wczytanie danych ===
df = pd.read_csv("mammography.csv", header=None)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values
y = LabelEncoder().fit_transform(y)  # -1 → 0, 1 → 1

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# === Model 1: XGBoost ===
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(objective="binary:logistic", max_depth=6, learning_rate=0.1,
                              n_estimators=200, scale_pos_weight=pos_weight)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds)
xgb_prec = precision_score(y_test, xgb_preds)
xgb_rec = recall_score(y_test, xgb_preds)

print("[XGBoost] Evaluation:")
print(f"Accuracy: {xgb_acc:.4f} | F1: {xgb_f1:.4f} | Precision: {xgb_prec:.4f} | Recall: {xgb_rec:.4f}")
print(classification_report(y_test, xgb_preds, digits=4))

# === Model 2: Cost-Sensitive PyTorch NN ===
class MammographyNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

class CombinedCostSensitiveLoss(nn.Module):
    def __init__(self, class_weights, reg_weight=0.5, boost_weight=1.0):
        super().__init__()
        self.class_weights = class_weights
        self.reg_weight = reg_weight
        self.boost_weight = boost_weight

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        base_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weights = targets * self.class_weights[1] + (1 - targets) * self.class_weights[0]
        weighted_loss = base_loss * weights
        margin_penalty = targets * torch.clamp(1 - probs, min=0)
        boost_factor = 1 + self.boost_weight * torch.abs(targets - probs)
        total_loss = (weighted_loss + self.reg_weight * margin_penalty) * boost_factor
        return total_loss.mean()

# Przygotowanie danych
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
test_ds = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# Trening modelu NN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MammographyNet(X.shape[1]).to(device)
counts = np.bincount(y_train)
weights = torch.tensor([len(y_train)/(2*c) for c in counts], dtype=torch.float32)
loss_fn = CombinedCostSensitiveLoss(weights, reg_weight=0.7, boost_weight=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

best_f1 = 0
model_path = "mammography_cost_sensitive_best.pt"

for epoch in range(50):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)

    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb))
            preds = (probs > 0.5).int()
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(yb.cpu().numpy())

    f1 = f1_score(labels_all, preds_all)
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), model_path)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | F1: {f1:.4f}")

# Ewaluacja końcowa
model.load_state_dict(torch.load(model_path))
model.eval()
preds_all, labels_all = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        probs = torch.sigmoid(model(xb))
        preds = (probs > 0.5).int()
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(yb.cpu().numpy())

nn_acc = accuracy_score(labels_all, preds_all)
nn_f1 = f1_score(labels_all, preds_all)
nn_prec = precision_score(labels_all, preds_all)
nn_rec = recall_score(labels_all, preds_all)

print("\n[NN Cost-sensitive] Evaluation:")
print(f"Accuracy: {nn_acc:.4f} | F1: {nn_f1:.4f} | Precision: {nn_prec:.4f} | Recall: {nn_rec:.4f}")
print("\nClassification Report:\n", classification_report(labels_all, preds_all, digits=4))

# Porównanie wykresowe
labels = ['Accuracy', 'F1-score', 'Precision', 'Recall']
xgb_scores = [xgb_acc, xgb_f1, xgb_prec, xgb_rec]
nn_scores = [nn_acc, nn_f1, nn_prec, nn_rec]

x = np.arange(len(labels))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, xgb_scores, width, label='XGBoost')
plt.bar(x + width/2, nn_scores, width, label='Cost-sensitive NN')
plt.ylabel('Score')
plt.title('Porównanie modeli - Mammography')
plt.xticks(x, labels)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# Predykcja na własnych danych
print("\nWprowadź dane wejściowe (6 liczb oddzielonych przecinkami):")
user_input = input("Przykład: 0.23,5.07,-0.27,0.83,-0.37,0.48\n> ")
vals = np.array([float(v.strip()) for v in user_input.split(",")]).reshape(1, -1)
vals_scaled = scaler.transform(vals)

# XGBoost
xgb_pred = xgb_model.predict(vals_scaled)[0]
print(f"[XGBoost] Predykcja: {'Rak (1)' if xgb_pred == 1 else 'Brak raka (0)'}")

# NN
tensor_input = torch.tensor(vals_scaled, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    output = torch.sigmoid(model(tensor_input))
    pred = (output > 0.5).int().item()
print(f"[NN] Predykcja: {'Rak (1)' if pred == 1 else 'Brak raka (0)'}")
