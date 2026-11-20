"""
Synthetic Patient Health Records generator (Practical-2: Data Cleanser)
-----------------------------------------------------------------------
Creates a realistic, reproducible dataset of 500 patients with
  * intentionally injected MISSING values  (age, gender, region, bmi, cholesterol, glucose)
  * intentionally injected OUTLIERS        (bmi, blood_pressure, cholesterol, glucose)

Outputs
  ../Data/Raw/patient_health_records_raw.csv          <- dataset given to students (dirty)
  ../Data/External/ground_truth_clean_values.csv      <- original values BEFORE corruption
                                                         (used only to *evaluate* imputation)
  ../Data/External/data_dictionary.csv
"""
import os
import numpy as np
import pandas as pd

SEED, N = 42, 500
rng = np.random.default_rng(SEED)

# ---------- 1. clean, correlated "true" patient data ----------
age = np.clip(rng.normal(52, 15, N), 18, 90).round().astype(int)
gender = rng.choice(["Male", "Female"], N, p=[0.52, 0.48])
region = rng.choice(["North", "South", "East", "West"], N, p=[0.34, 0.28, 0.22, 0.16])  # North is the mode
male = (gender == "Male").astype(int)

bmi = np.clip(rng.normal(26.5, 4.2, N) + 0.04 * (age - 52), 16.5, 42).round(1)
blood_pressure = np.clip(
    108 + 0.30 * (age - 52) + 0.9 * (bmi - 26.5) + 4 * male + rng.normal(0, 8, N) + 14, 90, 175
).round(1)
cholesterol = np.clip(
    185 + 0.9 * (age - 52) + 2.2 * (bmi - 26.5) + 6 * male + rng.normal(0, 28, N) + 8, 120, 320
).round(1)
glucose = np.clip(
    92 + 0.25 * (age - 52) + 1.6 * (bmi - 26.5) + rng.gamma(2.0, 4.0, N) - 8 + rng.normal(0, 7, N), 65, 190
).round(1)


def z(x):
    return (x - x.mean()) / x.std()


logit = -0.75 + 0.55 * z(age) + 0.65 * z(bmi) + 0.60 * z(blood_pressure) + 0.45 * z(cholesterol) + 0.60 * z(glucose)
disease_risk = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

patient_id = [f"PT{str(i).zfill(4)}" for i in range(1, N + 1)]
truth = pd.DataFrame(dict(
    patient_id=patient_id, age=age, gender=gender, region=region, bmi=bmi,
    blood_pressure=blood_pressure, cholesterol=cholesterol, glucose=glucose,
    disease_risk=disease_risk,
))

# ---------- 2. corrupt a copy: outliers first, then missing ----------
raw = truth.copy()
raw["age"] = raw["age"].astype(float)  # age becomes float while NaNs exist


def inject_outliers(col, n_high, n_low, high_rng, low_rng):
    idx = rng.choice(raw.index, n_high + n_low, replace=False)
    raw.loc[idx[:n_high], col] = rng.uniform(*high_rng, n_high).round(1)
    if n_low:
        raw.loc[idx[n_high:], col] = rng.uniform(*low_rng, n_low).round(1)
    return set(idx)


out_idx = {
    "bmi": inject_outliers("bmi", 7, 6, (52, 72), (8.5, 12.5)),                # very high / very low BMI
    "blood_pressure": inject_outliers("blood_pressure", 12, 0, (205, 270), (0, 0)),  # extreme high only
    "cholesterol": inject_outliers("cholesterol", 8, 5, (430, 620), (45, 85)),  # extreme low / high
    "glucose": inject_outliers("glucose", 12, 0, (320, 520), (0, 0)),           # very high spikes
}

miss_rate = {"age": 0.06, "gender": 0.05, "region": 0.05, "bmi": 0.08, "cholesterol": 0.07, "glucose": 0.07}
for col, p in miss_rate.items():
    candidates = [i for i in raw.index if i not in out_idx.get(col, set())]  # keep outliers visible
    idx = rng.choice(candidates, int(round(p * N)), replace=False)
    raw.loc[idx, col] = np.nan

# ---------- 3. save ----------
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")
raw.to_csv(os.path.join(base, "Raw", "patient_health_records_raw.csv"), index=False)
truth.to_csv(os.path.join(base, "External", "ground_truth_clean_values.csv"), index=False)

dd = pd.DataFrame([
    ["patient_id", "String", "Unique identifier for each patient", "No missing, no outliers"],
    ["age", "Integer", "Age of the patient (years)", "~6% missing"],
    ["gender", "Categorical", "Male / Female", "~5% missing"],
    ["region", "Categorical", "North / South / East / West", "~5% missing"],
    ["bmi", "Float", "Body Mass Index (kg/m^2)", "~8% missing + very high/low outliers"],
    ["blood_pressure", "Float", "Average systolic blood pressure (mmHg)", "Extreme high outliers"],
    ["cholesterol", "Float", "Cholesterol level (mg/dL)", "~7% missing + extreme low/high outliers"],
    ["glucose", "Float", "Fasting glucose level (mg/dL)", "~7% missing + very high spikes"],
    ["disease_risk", "Binary Int", "Target: 0 = Low risk, 1 = High risk", "No missing, used as target"],
], columns=["field_name", "data_type", "description", "missingness_or_outliers"])
dd.to_csv(os.path.join(base, "External", "data_dictionary.csv"), index=False)

if __name__ == "__main__":
    print("Raw shape:", raw.shape)
    print(raw.isna().sum())
    print(raw.describe().round(1))
    print(raw["disease_risk"].value_counts(normalize=True).round(3))
