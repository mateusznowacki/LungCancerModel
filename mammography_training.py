import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Wczytaj dane
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/mammography.csv"
df = pd.read_csv(url, header=None)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values
y = LabelEncoder().fit_transform(y)

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Dataset i DataLoader
class MammographyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MammographyDataset(X_train, y_train)
test_dataset = MammographyDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Definicja modelu
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 2)  # 2 klasy
        )

    def forward(self, x):
        return self.layers(x)

# Oblicz wagi klas
class_counts = np.bincount(y_train)
weights = 1.0 / class_counts
weights = torch.tensor(weights, dtype=torch.float32)

# Urządzenie
device = torch.device("cpu")  # wymuszenie CPU

# Inicjalizacja modelu
model = SimpleNN(input_size=X.shape[1]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening
model.train()
for epoch in range(20):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Ewaluacja
model.eval()
y_pred = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        preds = torch.argmax(outputs, dim=1)
        y_pred.extend(preds.cpu().numpy())

# Metryki
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))
