import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Ambil argumen dari MLflow atau CLI
parser = argparse.ArgumentParser()
parser.add_argument("n_estimators", type=int)
parser.add_argument("max_depth", type=int)
parser.add_argument("dataset", type=str)
args = parser.parse_args()

# Set tracking URI hanya jika disediakan lewat environment (misalnya lokal)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

# Set nama eksperimen
mlflow.set_experiment("Eksperimen klasifikasi berat badan")

# Baca dataset
data = pd.read_csv(args.dataset)
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Label", axis=1),
    data["Label"],
    test_size=0.2,
    random_state=42
)

# Aktifkan autolog
mlflow.sklearn.autolog()

# Latih model
model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
model.fit(X_train, y_train)

# Evaluasi akurasi
accuracy = model.score(X_test, y_test)
print(f"Akurasi: {accuracy}")

# Log model secara eksplisit agar bisa digunakan oleh mlflow models build-docker
mlflow.sklearn.log_model(model, artifact_path="model")
