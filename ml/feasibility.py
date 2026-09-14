"""
Deterministic Feasibility Calculator for BuildBusinessLK MSME Products.
Evaluates Capital Fit, Yield Fit, and Staffing Fit against domain benchmarks.
"""

from typing import Dict, Any

# Benchmarks for each product:
# (min_budget, target_budget, min_yield, target_yield, min_staff, target_staff)
PRODUCT_BENCHMARKS = {
    # Coconut
    "Coconut Chips": {
        "min_budget": 80000,
        "target_budget": 180000,
        "min_yield": 400,
        "target_yield": 1200,
        "min_staff": 1,
        "target_staff": 2,
    },
    "Coconut Flour": {
        "min_budget": 250000,
        "target_budget": 450000,
        "min_yield": 1000,
        "target_yield": 2500,
        "min_staff": 2,
        "target_staff": 3,
    },
    "Coconut Milk": {
        "min_budget": 350000,
        "target_budget": 750000,
        "min_yield": 1500,
        "target_yield": 3500,
        "min_staff": 2,
        "target_staff": 4,
    },
    "Virgin Coconut Oil": {
        "min_budget": 500000,
        "target_budget": 1200000,
        "min_yield": 1800,
        "target_yield": 4500,
        "min_staff": 3,
        "target_staff": 5,
    },
    "Desiccated Coconut": {
        "min_budget": 900000,
        "target_budget": 2200000,
        "min_yield": 4000,
        "target_yield": 9000,
        "min_staff": 4,
        "target_staff": 8,
    },

    # Kithul
    "Kithul Treacle": {
        "min_budget": 75000,
        "target_budget": 180000,
        "min_yield": 150,
        "target_yield": 600,
        "min_staff": 1,
        "target_staff": 2,
    },
    "Kithul Jaggery": {
        "min_budget": 150000,
        "target_budget": 320000,
        "min_yield": 300,
        "target_yield": 1000,
        "min_staff": 2,
        "target_staff": 3,
    },
    "Kithul Flour": {
        "min_budget": 280000,
        "target_budget": 550000,
        "min_yield": 500,
        "target_yield": 1600,
        "min_staff": 2,
        "target_staff": 4,
    },

    # Palmyrah
    "Palmyrah Jaggery": {
        "min_budget": 75000,
        "target_budget": 160000,
        "min_yield": 200,
        "target_yield": 700,
        "min_staff": 1,
        "target_staff": 2,
    },
    "Palmyrah Sugar": {
        "min_budget": 200000,
        "target_budget": 450000,
        "min_yield": 400,
        "target_yield": 1400,
        "min_staff": 2,
        "target_staff": 3,
    },
    "Palmyrah Treacle": {
        "min_budget": 150000,
        "target_budget": 320000,
        "min_yield": 350,
        "target_yield": 1100,
        "min_staff": 2,
        "target_staff": 3,
    },
}

DEFAULT_BENCHMARK = {
    "min_budget": 100000,
    "target_budget": 300000,
    "min_yield": 300,
    "target_yield": 1000,
    "min_staff": 1,
    "target_staff": 3,
}


def _calc_fit(actual: float, min_val: float, target_val: float) -> int:
    if actual <= 0:
        return 50  # baseline estimate if unspecified
    if actual < min_val:
        score = int(35 + (actual / min_val) * 35)
        return max(30, min(69, score))
    elif actual < target_val:
        fraction = (actual - min_val) / max(1, target_val - min_val)
        score = int(70 + fraction * 22)
        return max(70, min(92, score))
    else:
        # Exceeds target
        ratio = actual / target_val
        score = int(93 + min(6, (ratio - 1.0) * 3))
        return min(99, score)


def calculate_feasibility(profile: Dict[str, Any], product_name: str) -> Dict[str, int]:
    """
    Computes deterministic feasibility scores (0-100) for a product given a business profile.
    """
    benchmarks = PRODUCT_BENCHMARKS.get(product_name, DEFAULT_BENCHMARK)

    budget = float(profile.get("budget_lkr") or profile.get("budget") or 0)
    monthly_yield = float(profile.get("monthly_yield_kg") or profile.get("monthly_yield") or 0)
    employees = float(profile.get("employees") or 1)

    capital_fit = _calc_fit(budget, benchmarks["min_budget"], benchmarks["target_budget"])
    yield_fit = _calc_fit(monthly_yield, benchmarks["min_yield"], benchmarks["target_yield"])
    staffing_fit = _calc_fit(employees, benchmarks["min_staff"], benchmarks["target_staff"])

    return {
        "capitalFit": capital_fit,
        "yieldFit": yield_fit,
        "staffingFit": staffing_fit,
    }
