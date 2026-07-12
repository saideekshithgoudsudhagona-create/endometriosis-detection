"""
generate_data.py
-----------------
Creates a synthetic (but clinically plausible) endometriosis dataset so the
full pipeline can run end-to-end.

REPLACE THIS with your real dataset (CSV of patient records) once you have
one — just make sure the target column is named 'diagnosis' (1 = endometriosis,
0 = no endometriosis) and re-run train_model.py.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 800  # number of synthetic patients

age = np.random.randint(18, 45, N)
bmi = np.round(np.random.normal(24, 4, N), 1)
pelvic_pain_severity = np.random.randint(0, 11, N)          # 0-10 scale
cycle_irregularity = np.random.randint(0, 2, N)             # 0/1
infertility_history = np.random.randint(0, 2, N)            # 0/1
family_history = np.random.randint(0, 2, N)                 # 0/1
painful_intercourse = np.random.randint(0, 2, N)             # 0/1
heavy_bleeding = np.random.randint(0, 2, N)                  # 0/1
chronic_fatigue = np.random.randint(0, 2, N)                 # 0/1
ca125_level = np.round(np.random.normal(28, 12, N), 1)       # blood marker (U/mL)

# Build a "risk score" so the synthetic labels correlate realistically
# with known clinical risk factors (rather than being pure noise).
risk_score = (
    0.35 * (pelvic_pain_severity / 10)
    + 0.20 * cycle_irregularity
    + 0.15 * infertility_history
    + 0.10 * family_history
    + 0.10 * painful_intercourse
    + 0.05 * heavy_bleeding
    + 0.05 * chronic_fatigue
    + 0.15 * (ca125_level > 35).astype(int)
    + np.random.normal(0, 0.12, N)  # noise
)

# Only ~28% positive class -> realistic imbalance (mirrors real diagnostic rates)
threshold = np.quantile(risk_score, 0.72)
diagnosis = (risk_score > threshold).astype(int)

df = pd.DataFrame({
    "age": age,
    "bmi": bmi,
    "pelvic_pain_severity": pelvic_pain_severity,
    "cycle_irregularity": cycle_irregularity,
    "infertility_history": infertility_history,
    "family_history": family_history,
    "painful_intercourse": painful_intercourse,
    "heavy_bleeding": heavy_bleeding,
    "chronic_fatigue": chronic_fatigue,
    "ca125_level": ca125_level,
    "diagnosis": diagnosis,
})

df.to_csv("endometriosis_data.csv", index=False)
print(f"Saved endometriosis_data.csv with shape {df.shape}")
print(df["diagnosis"].value_counts(normalize=True))
