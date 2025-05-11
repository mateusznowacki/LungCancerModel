import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# tensorflow keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Loss
import tensorflow as tf

# === Wczytanie danych ===
df = pd.read_csv("dataset_prepared.csv")  # <- podmień na prawdziwy plik
df = df.drop(columns=["Patient Id", "Level"])  # kolumny nienumeryczne

# === Przygotowanie danych ===
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === MODEL 1: XGBoost klasyczny ===
xgb_model = xgb.XGBClassifier(objective="multi:softprob", num_class=num_classes)
xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
print(f"Model 1 - XGBoost accuracy: {xgb_acc:.4f}")

# === MODEL 2: Keras z Weighted CCE + cost-sensitive regularization ===

# Oblicz klasy i wagi
class_counts_arr = np.bincount(y_train)
total_samples = len(y_train)

class WeightedCategoricalCrossEntropy(Loss):
    def __init__(self, class_counts, regularization=True):
        super().__init__()
        self.class_counts = class_counts
        self.num_classes = len(class_counts)
        self.total_samples = np.sum(class_counts)
        self.weights = tf.constant(
            [self.total_samples / (self.num_classes * count) for count in class_counts],
            dtype=tf.float32
        )
        self.regularization = regularization

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=self.num_classes)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        base_loss = -tf.reduce_sum(self.weights * y_true_one_hot * tf.math.log(y_pred), axis=-1)

        if self.regularization:
            penalty = tf.reduce_sum(tf.square(y_pred - y_true_one_hot), axis=-1)
            return tf.reduce_mean(base_loss + 0.1 * penalty)
        return tf.reduce_mean(base_loss)

# Budowa modelu Keras
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Kompilacja modelu
loss_fn = WeightedCategoricalCrossEntropy(class_counts_arr, regularization=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Trenowanie
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

# Ewaluacja
keras_loss, keras_acc = model.evaluate(X_test, y_test)
print(f"Model 2 - Keras (Weighted CCE + reg) accuracy: {keras_acc:.4f}")
