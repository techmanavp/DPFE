<h1 align="center">Data Cleanser: Data Preprocessing & Feature Engineering</h1>

<p align="center">Practical-2 project: cleaning a healthcare dataset (patient health records) with missing values and outliers so it is ready for machine learning, with the goal of <b>predicting heart-disease risk</b>.</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Cleaning-purple" alt="Pandas">
  <img src="https://img.shields.io/badge/scikit--learn-Imputation-orange" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626" alt="Jupyter">
  <img src="https://img.shields.io/badge/Seaborn-Visualization-4c72b0" alt="Seaborn">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" alt="Status">
</p>

---

## 📌 Objective

Practise **data preprocessing and feature engineering** with a strong emphasis on **handling missing values** and **outlier detection / treatment**:

* Apply and compare six **imputation strategies** (Simple, Most-frequent, Missing-indicator + Random sample, KNN, MICE).
* Detect and treat outliers with **Z-score, IQR, Percentile and Winsorization**.
* Compare the dataset **before vs after** cleaning and deliver a **machine-learning-ready dataset** with a profiling report.

## 🗂 Dataset

500 synthetic patient records (generated reproducibly by `Notebooks/generate_dataset.py`, seed = 42). The *true* values were kept in `Data/External/ground_truth_clean_values.csv`, so every imputation / outlier method can be **scored exactly** against reality.

| Field | Type | Description | Issues injected |
|---|---|---|---|
| `patient_id` | String | Unique identifier | none |
| `age` | Integer | Age in years | 6 % missing |
| `gender` | Categorical | Male / Female | 5 % missing |
| `region` | Categorical | North / South / East / West | 5 % missing |
| `bmi` | Float | Body Mass Index | 8 % missing + 13 very high / low outliers |
| `blood_pressure` | Float | Systolic BP (mmHg) | 12 extreme high outliers |
| `cholesterol` | Float | mg/dL | 7 % missing + 13 extreme low / high outliers |
| `glucose` | Float | Fasting glucose (mg/dL) | 7 % missing + 12 very high spikes |
| `disease_risk` | Binary | 0 = Low, 1 = High | none (target, 39.4 % high risk) |

## 📁 Repository structure

```
Practical-2/
├── Data/
│   ├── Raw/              patient_health_records_raw.csv          ← dirty dataset (500 rows)
│   ├── External/         ground_truth_clean_values.csv, data_dictionary.csv
│   └── Processed/        imputed_*.csv, outliers_*.csv, cleaned_patient_health_records.csv,
│                         cleaned_with_engineered_features.csv, ml_ready_features.csv,
│                         comparison tables (*.csv) and results_summary.json
├── Notebooks/
│   ├── Part-A.ipynb                                           ← missing values (Tasks 1-2)
│   ├── Part-B.ipynb                                           ← outliers (Tasks 3-5)
│   ├── Practical2_Data_Preprocessing_Feature_Engineering.ipynb ← full pipeline A + B + C (Tasks 1-7)
│   ├── generate_dataset.py                                    ← creates the raw dataset
│   └── build_reports.py                                       ← creates the HTML / PDF reports
├── Report/
│   ├── cleaned_profiling_report.html                          ← data-profiling report
│   ├── Brief_Report.pdf                                       ← Task 7 brief report
│   └── Theory_Concepts.pdf                                    ← definitions of all concepts used
├── Visualization/        01 … 14 PNG charts (missingness, imputation, outliers, before/after, models)
├── requirements.txt
└── README.md
```

## 🔬 Methodology

| Step | What was done |
|---|---|
| **Part A – missing values** | Missing-value report (190 cells = 4.22 %, 32.8 % of rows) · Mean / Median · Most-frequent (region, gender) · Missing-indicator + Random sample · KNN (k = 5) · MICE (Bayesian Ridge) · scored against true values |
| **Part B – outliers** | Z-score (cholesterol, glucose) · IQR (BMI) · Percentile cap 1 %/99 % · Winsorization 5 % · modified Z-score (bonus) · all treatments compared on rows kept, error vs truth, remaining extreme cells and ROC-AUC |
| **Part C – final dataset** | Order-of-operations experiment · final cleaned dataset · feature engineering (age groups, WHO BMI classes, BP stages, cholesterol / glucose bands, metabolic-flag count, encoding, scaling) · quality scorecard · Logistic Regression & Random Forest check |

## 📊 Key results

**Imputation (normalised RMSE vs the true values, lower = better)**

| Technique | Raw data (outliers present) | After outliers are flagged |
|---|---|---|
| Mean + Most-frequent | 1.127 | 1.226 |
| Median + Most-frequent | **1.026** | 1.227 |
| Missing-indicator + Random sample | 1.757 | – |
| KNN (k = 5) | 1.553 | 1.126 |
| MICE (Bayesian Ridge) | 1.069 | **0.979** |

**Outlier treatment**

| Treatment | Rows kept | Error vs truth | Extreme cells left | ROC-AUC |
|---|---|---|---|---|
| No treatment | 500 | 2.128 | 50 | 0.833 |
| Z-score removal (\|z\|>3) | 462 | 0.316 | 11 | 0.884 |
| IQR removal | 438 | 0.221 | 0 | 0.873 |
| Percentile cap 1 %–99 % | 500 | 1.940 | 50 | 0.839 |
| **Winsorization 5 %** | **500** | 0.404 | **0** | 0.875 |
| IQR capping | 500 | 0.477 | 11 | 0.870 |

**Final pipeline:** IQR-flagged outliers → treated as missing → **MICE** imputes every gap (error vs truth **0.266**, 28 % lower than impute-then-cap).

**Before vs after**

| Indicator | Raw | Cleaned |
|---|---|---|
| Missing cells | 190 | **0** |
| Implausible values | 50 | **0** |
| Avg \|skewness\| of 4 columns | 4.27 | **0.04** |
| Max glucose / cholesterol / BP | 499 / 618 / 265 | 120 / 284 / 149 |
| Logistic Regression ROC-AUC (accuracy) | 0.831 (74.1 %) | **0.874 (79.3 %)** |
| Random Forest ROC-AUC (accuracy) | 0.854 (76.8 %) | **0.862 (78.5 %)** |

## ✅ Conclusions

1. **Most effective imputation:** median when outliers are still present; **MICE** once they have been neutralised. Mode is sufficient for gender / region.
2. **Best outlier handling for data quality:** **Winsorization** (no data loss, no extreme value left); treating implausible values as missing + MICE gave the lowest error of all.
3. **Usability:** a complete, outlier-free, correctly-scaled dataset in which the relationships with `disease_risk` are restored and linear models gain about 4 AUC points.

<div align="center">

## 👨‍💻 Author

**Manav Patel**
📧 manavpatel.tech@gmail.com
🔗 [GitHub](https://github.com/techmanavp) 

</div>