import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "business_training.csv"
MODELS_DIR = ROOT_DIR.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "recommendation_model.pkl"
SECTOR_ENCODER_PATH = MODELS_DIR / "sector_encoder.pkl"
EXPERIENCE_ENCODER_PATH = MODELS_DIR / "experience_encoder.pkl"
TARGET_ENCODER_PATH = MODELS_DIR / "target_encoder.pkl"


def train_and_save_model() -> None:
    df = pd.read_csv(DATA_PATH)

    sector_encoder = LabelEncoder()
    experience_encoder = LabelEncoder()
    target_encoder = LabelEncoder()

    df["sector"] = sector_encoder.fit_transform(df["sector"])
    df["experience"] = experience_encoder.fit_transform(df["experience"])
    y = target_encoder.fit_transform(df["recommendation"])

    X = df[
        [
            "sector",
            "budget",
            "monthly_yield",
            "employees",
            "experience",
        ]
    ]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(sector_encoder, SECTOR_ENCODER_PATH)
    joblib.dump(experience_encoder, EXPERIENCE_ENCODER_PATH)
    joblib.dump(target_encoder, TARGET_ENCODER_PATH)

    print(f"Model trained and saved to {MODELS_DIR}")


if __name__ == "__main__":
    train_and_save_model()