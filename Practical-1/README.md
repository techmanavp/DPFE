<div align="center">

# Data Preprocessing & Feature Engineering: Customer Churn

Practical-1 project: preparing an e-commerce customer dataset for machine learning, with the goal of **predicting customer churn**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Cleaning-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0)
![Status](https://img.shields.io/badge/Status-Completed-2ea44f)

</div>

---

## 📑 Table of Contents
[Overview](#-overview) · [Problem Statement](#-problem-statement) · [Dataset](#-dataset) · [Workflow](#-workflow) · [Visual Results](#-visual-results) · [Key Insights](#-key-insights) · [Repo Structure](#-repository-structure) · [Author]

---

## 🔍 Overview

An online store is losing customers (**churn**). Before any model can predict who will leave, the raw data has to be **loaded from different sources, cleaned, understood and transformed**. This project does exactly that on a real e-commerce churn dataset of **5,630 customers**.


| 📦 Rows | 🧱 Raw columns | 🧹 Missing cells fixed | 🧬 ML-ready columns | 🎯 Churn rate |
|:---:|:---:|:---:|:---:|:---:|
| **5,630** | **20** | **1,856** | **34** | **~17%** |

---

## 🎯 Problem Statement

| | |
|---|---|
| **Business problem** | Customers leave the platform, reducing revenue. Keeping a customer costs less than winning a new one. |
| **ML task** | Supervised learning, **binary classification** |
| **Target (y)** | `Churn`: `0` = stays, `1` = churns |
| **Features (X)** | Tenure, order count, cashback, days since last order, complaints, satisfaction, gender, age, income and more |
| **Metrics** | Recall, Precision, F1-score, ROC-AUC (plain accuracy is misleading because only ~17% churn) |

---

## 🗂️ Dataset

**E-commerce Customer Churn dataset** from Kaggle: `<PASTE KAGGLE DATASET LINK HERE>`

| Group | Columns |
|---|---|
| 👤 Profile | `Gender`, `MaritalStatus`, `CityTier`, `NumberOfAddress` |
| 🛍️ Purchase behaviour | `OrderCount`, `CouponUsed`, `CashbackAmount`, `OrderAmountHikeFromlastYear`, `PreferedOrderCat`, `DaySinceLastOrder` |
| 📱 Engagement | `Tenure`, `PreferredLoginDevice`, `HourSpendOnApp`, `NumberOfDeviceRegistered`, `PreferredPaymentMode`, `WarehouseToHome` |
| 😊 Feedback | `SatisfactionScore`, `Complain` |
| 🎯 Target | `Churn` |

---

## 🔄 Workflow

```mermaid
flowchart LR
    A["CSV file"] --> M
    B["JSON file"] --> M
    C["SQLite database"] --> M
    M["Merge<br/>5,630 rows"] --> D["Clean<br/>missing, labels, outliers"]
    D --> F["Feature engineering<br/>new features, encoding, scaling"]
    F --> E["EDA<br/>uni, bi, multivariate"]
    F --> R[("ML-ready data")]
    E --> P["Profiling reports"]
```

---

## 🖼️ Visual Results

### Outlier handling
![Outlier capping](Visualization/01_outlier_capping.png)

### Univariate analysis
![Age, Income, Purchases](Visualization/02_univariate_age_income_purchases.png)
![Tenure, Cashback, Churn](Visualization/03_univariate_tenure_cashback_churn.png)

> Purchases are strongly right-skewed (median 2 orders). Only about **1 in 6 customers churns**, so the target is imbalanced.

### Bivariate analysis
![Gender and Income](Visualization/04_bivariate_gender_income.png)
![Churn drivers](Visualization/05_bivariate_churn_drivers.png)

### Multivariate analysis
![Correlation heatmap](Visualization/06_correlation_heatmap.png)
![Pair plot](Visualization/07_pairplot.png)

---

## 🔑 Key Insights

| # | Finding | Evidence |
|:-:|---|---|
| 1 | **New customers are the biggest risk** | Churned customers average **3.9 months** of tenure vs **11.4** for retained ones |
| 2 | **Complaints triple the churn rate** | **31.7%** churn with a complaint vs **10.9%** without |
| 3 | **Single customers churn the most** | Single **26.7%**, Divorced 14.6%, Married 11.5% |
| 4 | **Payment mode matters** | Cash on Delivery **24.9%**, E-wallet **22.8%** vs Credit Card **14.2%** |
| 5 | **Lower cashback goes with churn** | Average cashback **160** (churned) vs **181** (retained) |
| 6 | **Gender barely affects purchases** | Mean orders 3.06 (F) vs 2.90 (M), identical medians |
| 7 | **Strongest correlations with churn** | Tenure **-0.34**, Complain **+0.25** |

---

## 📁 Repository Structure

```
Practical-1/
├── 📄 README.md
├── 📂 Data/
│   ├── 📂 Raw/          original dataset + CSV / JSON / SQLite parts (untouched)
│   ├── 📂 External/     data fetched from the API
│   └── 📂 Processed/    cleaned_churn_data.csv, ml_ready_churn_data.csv
├── 📂 Notebooks/
│   ├── 📓 Part-A.ipynb                        theory and tensors
│   └── 📓 Practical1_Data_Preprocessing.ipynb acquisition → cleaning → EDA → profiling
├── 📂 Report/
│   └── 🌐 cleaned_profiling_report.html
└── 📂 Visualization/    7 saved charts (PNG)
```

| Folder | Meaning |
|---|---|
| `Raw` | Data exactly as received. Never edited, so the pipeline can always be re-run from the start |
| `External` | Data that comes from outside the main dataset (API) |
| `Processed` | Final cleaned and ML-ready data |

---

<div align="center">

## 👨‍💻 Author

**Manav Patel**
📧 manavpatel.tech@gmail.com
🔗 [GitHub](https://github.com/techmanavp) 

</div>