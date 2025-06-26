# Import library
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Ambil nilai dari environment variable (jika ada)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

# Set nama eksperimen
mlflow.set_experiment("Eksperimen klasifikasi berat badan")

# Baca data
data = pd.read_csv("Obesity Classification Preprocessing.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Label", axis=1),
    data["Label"],
    random_state=42,
    test_size=0.2
)

# Input contoh untuk autologging (opsional)
input_example = X_train[0:5]

# Aktifkan autologging
mlflow.sklearn.autolog()

# Mulai run
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=505, max_depth=37)
    model.fit(X_train, y_train)

    # Evaluasi
    accuracy = model.score(X_test, y_test)
    print("Accuracy:", accuracy)
