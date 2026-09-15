from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import pandas as pd

from ml.feasibility import calculate_feasibility

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "product_recommender_v1.joblib"
# Legacy fallback path
LEGACY_PATH = ROOT_DIR.parent / "models" / "product_recommender_v1.joblib"

MODEL_VERSION = "product-recommender-v1"

_PIPELINE = None


def load_model():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    try:
        if MODEL_PATH.exists():
            _PIPELINE = joblib.load(MODEL_PATH)
        elif LEGACY_PATH.exists():
            _PIPELINE = joblib.load(LEGACY_PATH)
    except Exception as e:
        print(f"Warning: Failed to load ML model: {e}")
        _PIPELINE = None
    return _PIPELINE


# Initialize on import
load_model()


def normalize_sector(sector: Any) -> str:
    s = str(sector or "").strip().lower()
    if "palmyr" in s or "thal" in s:
        return "palmyrah"
    if "kithul" in s or "kitul" in s:
        return "kithul"
    if "coconut" in s or "coco" in s or "pol" in s:
        return "coconut"
    return "coconut"



def normalize_experience(exp: Any) -> float:
    if isinstance(exp, (int, float)):
        return max(0.0, float(exp))
    s = str(exp or "").strip().lower()
    if "beginner" in s or "no exp" in s or "new" in s or "none" in s:
        return 0.5
    if "advanced" in s or "expert" in s or "senior" in s or "5+" in s:
        return 5.0
    if "intermediate" in s or "some" in s or "few" in s:
        return 2.5
    try:
        return float(s)
    except ValueError:
        return 2.0


def recommend_products(features: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
    """
    Canonical ML product recommendation function.
    Returns Top-K products with confidence scores and deterministic feasibility breakdown.
    """
    model = load_model()
    if model is None:
        raise RuntimeError("ML recommendation model is not loaded.")

    sector = normalize_sector(features.get("sector"))
    budget = float(features.get("budget_lkr") or features.get("budget") or 250000)
    monthly_yield = float(features.get("monthly_yield_kg") or features.get("monthly_yield") or 1000)
    employees = int(features.get("employees") or 2)
    experience = normalize_experience(features.get("experience_years") or features.get("experience"))

    input_df = pd.DataFrame([
        {
            "sector": sector,
            "budget_lkr": budget,
            "monthly_yield_kg": monthly_yield,
            "employees": employees,
            "experience_years": experience,
        }
    ])

    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    # Filter by selected sector if applicable (only recommend products of the sector)
    sector_prefix_map = {
        "coconut": ["Coconut", "Virgin Coconut", "Desiccated"],
        "kithul": ["Kithul"],
        "palmyrah": ["Palmyrah", "Palm"],
    }
    allowed_prefixes = sector_prefix_map.get(sector, [])

    ranked_indices = probabilities.argsort()[::-1]
    recommendations: List[Dict[str, Any]] = []

    for idx in ranked_indices:
        prod_name = str(classes[idx])
        prob = float(probabilities[idx])
        # Filter to matched sector products
        if allowed_prefixes and not any(prod_name.startswith(p) for p in allowed_prefixes):
            continue
        rank = len(recommendations) + 1
        recommendations.append({
            "rank": rank,
            "product": prod_name,
            "confidence": round(prob * 100.0, 1),
        })
        if len(recommendations) >= top_k:
            break

    # If all sector-filtered probabilities were 0 or none matched, fallback to top classes without filter
    if not recommendations:
        for i, idx in enumerate(ranked_indices[:top_k]):
            recommendations.append({
                "rank": i + 1,
                "product": str(classes[idx]),
                "confidence": round(float(probabilities[idx]) * 100.0, 1),
            })

    top_product = recommendations[0]["product"] if recommendations else "Value-Added Production"
    feasibility = calculate_feasibility(
        {
            "budget_lkr": budget,
            "monthly_yield_kg": monthly_yield,
            "employees": employees,
        },
        top_product,
    )

    return {
        "modelVersion": MODEL_VERSION,
        "recommendations": recommendations,
        "feasibility": feasibility,
    }


def recommend_business(
    sector: str,
    budget: int,
    monthly_yield: int,
    employees: int,
    experience: str,
) -> str:
    """
    Backward-compatible helper returning the top recommended product name.
    """
    res = recommend_products({
        "sector": sector,
        "budget_lkr": budget,
        "monthly_yield_kg": monthly_yield,
        "employees": employees,
        "experience_years": experience,
    }, top_k=1)
    recs = res.get("recommendations", [])
    return recs[0]["product"] if recs else "Value-Added Production"
