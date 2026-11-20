from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    df = pd.read_csv("data/raw/transaction.csv")

    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    train , test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    train.to_csv(
        "data/processed/train.csv", index=False
    )
    test.to_csv(
        "data/processed/test.csv", index=False
    )

    print("Data Prepration Completed!")

if __name__ == "__main__":
    prepare_data()