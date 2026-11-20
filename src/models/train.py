
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

FEATURES = [
    "amount",
    "distance",
    "hour",
    "frequency",
    "account_age",
    "merchant_risk",
    "device_score",
    "location_score",
    "velocity",
    "previous_frauds"
]


def train():

    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Load datasets
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df[FEATURES]
    y_train = train_df["label"]

    X_test = test_df[FEATURES]
    y_test = test_df["label"]

    # Model parameters
    model_params = params["model"]
    threshold = params["training"]["threshold"]
    minimum_f1 = params["metrics"]["minimum_f1"]

    # Initialize model
    model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        random_state=model_params["random_state"],
        class_weight="balanced",
        n_jobs=-1
    )

    # MLflow configuration
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("fraud_detection")

    with mlflow.start_run():

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),

            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),

            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),

            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),

            "roc_auc": roc_auc_score(
                y_test, probabilities
            )
        }

        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("threshold", threshold)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Quality check
        if metrics["f1"] < minimum_f1:

            mlflow.set_tag("quality_check", "failed")

            raise ValueError(
                f"F1 score is below required threshold: "
                f"{metrics['f1']:.4f} "
                f"(minimum: {minimum_f1})"
            )

        # Create models directory
        Path("models").mkdir(exist_ok=True)

        # Save model locally
        joblib.dump(model, "models/model.joblib")

        # Log model in MLflow
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train.head(5),
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # Log configuration
        mlflow.log_artifact("params.yaml")

        # Mark successful training
        mlflow.set_tag("quality_check", "passed")

        print("Training Completed Successfully!")
        print("Metrics:")
        print(metrics)


if __name__ == "__main__":
    train()
