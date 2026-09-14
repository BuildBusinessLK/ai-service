import random
from pathlib import Path
import pandas as pd

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "ml" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Product archetypes with realistic Sri Lankan parameters:
# (sector, product, budget_min, budget_max, yield_min, yield_max, emp_min, emp_max, exp_min, exp_max)
ARCHETYPES = [
    # Coconut Sector
    ("coconut", "Coconut Chips", 80000, 320000, 400, 2200, 1, 3, 0.0, 3.0),
    ("coconut", "Coconut Flour", 250000, 700000, 1000, 4500, 2, 4, 1.0, 5.0),
    ("coconut", "Coconut Milk", 350000, 1200000, 1500, 6000, 2, 5, 1.0, 6.0),
    ("coconut", "Virgin Coconut Oil", 500000, 2200000, 1800, 8000, 3, 7, 1.5, 8.0),
    ("coconut", "Desiccated Coconut", 900000, 4000000, 4000, 15000, 4, 12, 2.0, 10.0),

    # Kithul Sector
    ("kithul", "Kithul Treacle", 75000, 350000, 150, 1200, 1, 3, 0.0, 5.0),
    ("kithul", "Kithul Jaggery", 150000, 550000, 300, 2200, 2, 4, 1.0, 7.0),
    ("kithul", "Kithul Flour", 280000, 850000, 500, 3000, 2, 5, 1.0, 7.0),

    # Palmyrah Sector
    ("palmyrah", "Palmyrah Jaggery", 75000, 320000, 200, 1500, 1, 3, 0.0, 5.0),
    ("palmyrah", "Palmyrah Sugar", 200000, 650000, 400, 2500, 2, 4, 1.0, 7.0),
    ("palmyrah", "Palmyrah Treacle", 150000, 500000, 350, 2200, 2, 4, 1.0, 6.0),
]

def generate_dataset(samples_per_product: int = 18):
    rows = []
    for sector, product, b_min, b_max, y_min, y_max, e_min, e_max, exp_min, exp_max in ARCHETYPES:
        for _ in range(samples_per_product):
            # Budget with small step rounding
            raw_b = random.uniform(b_min, b_max)
            budget = int(round(raw_b / 5000.0) * 5000)
            
            # Yield with small step rounding
            raw_y = random.uniform(y_min, y_max)
            yield_val = int(round(raw_y / 50.0) * 50)
            
            # Employees
            employees = random.randint(e_min, e_max)
            
            # Experience in years
            experience = round(random.uniform(exp_min, exp_max), 1)
            
            rows.append({
                "sector": sector,
                "budget_lkr": budget,
                "monthly_yield_kg": yield_val,
                "employees": employees,
                "experience_years": experience,
                "product": product,
            })
            
    df = pd.DataFrame(rows)
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    out_path = DATA_DIR / "business_training_v1.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} training samples across {len(ARCHETYPES)} products in {out_path}")
    print(df["product"].value_counts())

if __name__ == "__main__":
    generate_dataset(samples_per_product=20)
