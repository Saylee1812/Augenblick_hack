import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# ✅ Step 1: Load Dataset
df_train = pd.read_csv("data/train_FD001.txt", sep="\\s+")
df_train.columns = ["engine", "cycle"] + [f"sensor_{i}" for i in range(1, df_train.shape[1] - 1)]

# ✅ Step 2: Data Preprocessing
# Drop engine & cycle columns
X = df_train.drop(columns=["engine", "cycle"]).values

# Create anomaly labels (1 = anomaly, 0 = normal) based on sensor mean
y = (X.mean(axis=1) > X.mean().mean()).astype(int)

# Standardize data (for better model performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Step 4: Train & Save Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "models/random_forest.pkl")
print("✅ Random Forest Model Saved!")

# ✅ Step 5: Train & Save KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
joblib.dump(knn_model, "models/knn_model.pkl")
print("✅ KNN Model Saved!")

# ✅ Step 6: Train & Save SVM Model
svm_model = SVC(kernel="rbf", probability=True)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, "models/svm_model.pkl")
print("✅ SVM Model Saved!")

# ✅ Step 7: Train & Save LSTM Model
# Reshape for LSTM (samples, time steps, features)
X_train_LSTM = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_LSTM = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(1, activation="sigmoid")
])

lstm_model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
lstm_model.fit(X_train_LSTM, y_train, epochs=10, batch_size=32, validation_data=(X_test_LSTM, y_test))

# Save LSTM model
lstm_model.save("models/lstm_model.h5")
print("✅ LSTM Model Saved!")
