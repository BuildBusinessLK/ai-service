from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR.parent / "models"
MODEL_PATH = MODELS_DIR / "recommendation_model.pkl"
SECTOR_ENCODER_PATH = MODELS_DIR / "sector_encoder.pkl"
EXPERIENCE_ENCODER_PATH = MODELS_DIR / "experience_encoder.pkl"
TARGET_ENCODER_PATH = MODELS_DIR / "target_encoder.pkl"

_model = None
_sector_encoder = None
_experience_encoder = None
_target_encoder = None
_loaded = False


def _load_artifacts():
    global _model, _sector_encoder, _experience_encoder, _target_encoder, _loaded
    if _loaded:
        return
    try:
        import joblib
        if MODEL_PATH.exists() and SECTOR_ENCODER_PATH.exists():
            _model = joblib.load(MODEL_PATH)
            _sector_encoder = joblib.load(SECTOR_ENCODER_PATH)
            _experience_encoder = joblib.load(EXPERIENCE_ENCODER_PATH)
            _target_encoder = joblib.load(TARGET_ENCODER_PATH)
    except Exception:
        pass
    _loaded = True


def _normalize_string(value: Any) -> str:
    val = str(value or "").strip().lower()
    if val == "palmyra":
        return "palmyrah"
    return val


def _rule_based_recommendation(sector: str, budget: int, monthly_yield: int, employees: int, experience: str) -> str:
    s = _normalize_string(sector)
    b = int(budget) if budget else 250000

    if "coconut" in s:
        if b < 250000:
            return "Coconut Chips"
        elif b < 500000:
            return "Coconut Flour"
        elif b < 1000000:
            return "Virgin Coconut Oil"
        else:
            return "Desiccated Coconut"
    elif "palmyr" in s:
        if b < 250000:
            return "Palm Jaggery"
        elif b < 400000:
            return "Palm Sugar"
        else:
            return "Palm Treacle"
    elif "kithul" in s:
        if b < 200000:
            return "Kithul Treacle"
        elif b < 350000:
            return "Kithul Jaggery"
        else:
            return "Kithul Flour"
    return "Value-Added Production"


def recommend_business(
    sector: str,
    budget: int,
    monthly_yield: int,
    employees: int,
    experience: str,
) -> str:
    _load_artifacts()
    normalized_sector = _normalize_string(sector)
    normalized_experience = _normalize_string(experience)

    if _model is not None and _sector_encoder is not None:
        try:
            import pandas as pd
            if (
                normalized_sector in _sector_encoder.classes_
                and normalized_experience in _experience_encoder.classes_
            ):
                data = pd.DataFrame([
                    {
                        "sector": _sector_encoder.transform([normalized_sector])[0],
                        "budget": int(budget),
                        "monthly_yield": int(monthly_yield),
                        "employees": int(employees),
                        "experience": _experience_encoder.transform([normalized_experience])[0],
                    }
                ])
                prediction = _model.predict(data)
                return _target_encoder.inverse_transform(prediction)[0]
        except Exception:
            pass

    return _rule_based_recommendation(normalized_sector, budget, monthly_yield, employees, normalized_experience)

