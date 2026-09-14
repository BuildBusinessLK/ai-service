import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "business_training_v1.csv"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Also maintain root models dir for legacy compatibility
LEGACY_MODELS_DIR = ROOT_DIR.parent / "models"
LEGACY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "product-recommender-v1"
MODEL_PATH = MODELS_DIR / "product_recommender_v1.joblib"
METRICS_PATH = MODELS_DIR / "metrics_v1.json"


def train_and_evaluate_pipeline():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples from {DATA_PATH}")

    categorical_features = ["sector"]
    numerical_features = [
        "budget_lkr",
        "monthly_yield_kg",
        "employees",
        "experience_years",
    ]

    X = df[categorical_features + numerical_features]
    y = df["product"]

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_features,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate on Test Set
    test_preds = pipeline.predict(X_test)
    test_probs = pipeline.predict_proba(X_test)
    classes = pipeline.classes_

    top1_acc = float(np.mean(test_preds == y_test))

    # Top-3 Accuracy
    top3_correct = 0
    for idx, actual in enumerate(y_test):
        top3_indices = test_probs[idx].argsort()[::-1][:3]
        top3_classes = classes[top3_indices]
        if actual in top3_classes:
            top3_correct += 1
    top3_acc = float(top3_correct / len(y_test))

    macro_f1 = float(f1_score(y_test, test_preds, average="macro"))

    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, test_preds))

    # Cross validation for generalizability
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        p = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
        p.fit(X_tr, y_tr)
        cv_scores.append(float(p.score(X_val, y_val)))

    metrics = {
        "model_version": MODEL_VERSION,
        "dataset_version": "business-training-v1",
        "training_samples": len(df),
        "features": categorical_features + numerical_features,
        "classes": list(classes),
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "cv_mean_accuracy": round(float(np.mean(cv_scores)), 4),
        "cv_std_accuracy": round(float(np.std(cv_scores)), 4),
    }

    # Save Pipeline and Metrics
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(pipeline, LEGACY_MODELS_DIR / "product_recommender_v1.joblib")
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model successfully saved to {MODEL_PATH}")
    print(f"Metrics exported to {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train_and_evaluate_pipeline()