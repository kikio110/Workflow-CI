import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import numpy as np
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Ambil parameter dari sys.argv (bisa dari CLI atau MLflow CLI)
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37
    dataset_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")

    # Load data
    data = pd.read_csv(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Label", axis=1),
        data["Label"],
        test_size=0.2,
        random_state=42
    )

    # Input example untuk log_model
    input_example = X_train.head(5)

    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

        # Logging model dan metrik
    mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
    mlflow.log_metric("accuracy", accuracy)

    print(f"Akurasi: {accuracy}")
