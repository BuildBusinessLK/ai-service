from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR.parent / "models"
MODEL_PATH = MODELS_DIR / "recommendation_model.pkl"
SECTOR_ENCODER_PATH = MODELS_DIR / "sector_encoder.pkl"
EXPERIENCE_ENCODER_PATH = MODELS_DIR / "experience_encoder.pkl"
TARGET_ENCODER_PATH = MODELS_DIR / "target_encoder.pkl"

model = joblib.load(MODEL_PATH)
sector_encoder = joblib.load(SECTOR_ENCODER_PATH)
experience_encoder = joblib.load(EXPERIENCE_ENCODER_PATH)
target_encoder = joblib.load(TARGET_ENCODER_PATH)


def _normalize_string(value: Any) -> str:
    return str(value).strip().lower()


def recommend_business(
    sector: str,
    budget: int,
    monthly_yield: int,
    employees: int,
    experience: str,
) -> str:
    normalized_sector = _normalize_string(sector)
    normalized_experience = _normalize_string(experience)

    if normalized_sector not in sector_encoder.classes_:
        raise ValueError(
            f"Unsupported sector '{sector}'. Supported values: {', '.join(sector_encoder.classes_)}."
        )

    if normalized_experience not in experience_encoder.classes_:
        raise ValueError(
            f"Unsupported experience '{experience}'. Supported values: {', '.join(experience_encoder.classes_)}."
        )

    data = pd.DataFrame([
        {
            "sector": sector_encoder.transform([normalized_sector])[0],
            "budget": int(budget),
            "monthly_yield": int(monthly_yield),
            "employees": int(employees),
            "experience": experience_encoder.transform([normalized_experience])[0],
        }
    ])

    prediction = model.predict(data)
    return target_encoder.inverse_transform(prediction)[0]
