# BuildBusinessLK ML Dataset Documentation

## Overview
This directory stores versioned datasets for training the BuildBusinessLK Product Recommendation model.

## Dataset: `business_training_v1.csv`
- **Version**: `business-training-v1`
- **Sectors Covered**:
  - `coconut`
  - `kithul`
  - `palmyrah`
- **Target Variable**: `product`
- **Candidate Products**:
  - **Coconut**: Virgin Coconut Oil, Coconut Flour, Coconut Chips, Desiccated Coconut, Coconut Milk
  - **Kithul**: Kithul Treacle, Kithul Jaggery, Kithul Flour
  - **Palmyrah**: Palmyrah Jaggery, Palmyrah Sugar, Palmyrah Treacle

## Feature Definitions
| Feature | Type | Description | Plausible Sri Lankan Range |
| :--- | :--- | :--- | :--- |
| `sector` | Categorical | Target value chain (`coconut`, `kithul`, `palmyrah`) | N/A |
| `budget_lkr` | Numerical | Total starting capital in Sri Lankan Rupees (LKR) | 75,000 – 4,000,000 |
| `monthly_yield_kg` | Numerical | Monthly raw material or input availability (kg or liters) | 150 – 15,000 |
| `employees` | Numerical | Number of active production workers | 1 – 12 |
| `experience_years` | Numerical | Years of practical agro-processing experience | 0 – 10 |

## Grounding & Assumptions
1. **Low Capital Entry (LKR < 250k)**: Best suited for artisanal processing requiring minimal machinery: Coconut Chips (slicer + dehydrator), Kithul Treacle (boiling pans, refractometer, bottling), and Palm Jaggery (traditional molds).
2. **Intermediate Capital (LKR 250k – 750k)**: Suitable for semi-mechanized production: Coconut Flour (pulverizers, sifters), Kithul Flour (pith extraction, washing, drying), Palm Sugar (crystallization, centrifugal separator), Coconut Milk (hydraulic/screw press, pasteurizer).
3. **High Capital (LKR > 750k)**: Demands commercial machinery and food safety compliance: Virgin Coconut Oil (cold press, stainless centrifugation, micro-filtration), Desiccated Coconut (industrial shredders, steam blancher, fluid bed dryer).
