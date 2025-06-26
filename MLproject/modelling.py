import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

if __name__ == "__main__":
    # Supress warning
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Ambil argumen dari CLI atau MLflow
    parser = argparse.ArgumentParser()
    parser.add_argument("n_estimators", type=int, nargs="?", default=505)
    parser.add_argument("max_depth", type=int, nargs="?", default=37)
    parser.add_argument("dataset", type=str, nargs="?", default="MLproject/Obesity Classification Preprocessing.csv")
    args = parser.parse_args()

    # Set tracking URI jika disediakan (untuk local / server)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Set nama eksperimen
    mlflow.set_experiment("Eksperimen klasifikasi berat badan")

    # Load dataset
    data = pd.read_csv(args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Label", axis=1),
        data["Label"],
        test_size=0.2,
        random_state=42
    )
    input_example = X_train[:5]

    # Mulai run MLflow
    with mlflow.start_run():
        # Buat model
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
        model.fit(X_train, y_train)

        # Prediksi
        predicted = model.predict(X_test)

        # Log model ke artifact path 'model'
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # Log metrik evaluasi
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        print(f"Akurasi: {accuracy}")
