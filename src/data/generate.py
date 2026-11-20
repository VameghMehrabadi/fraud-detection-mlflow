from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification

def generate_data():
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.98,0.02],
        flip_y=0.01,
        random_state=42
    )
    columns = [
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

    df = pd.DataFrame(X, columns=columns)

    df["amount"] = abs(df["amount"])*100
    df["distance"] = abs(df["distance"])*100
    df["hour"]= (abs(df["hour"])* 12).astype(int) % 24
    df["label"]= y

    output = Path("data/raw")
    output.mkdir(parents=True, exist_ok=True)

    df.to_csv(output / "transaction.csv", index=False)

    print(f"Dataset Created : {len(df)} rows")
    print(df["label"].value_counts())

if __name__ == "__main__":
    generate_data()