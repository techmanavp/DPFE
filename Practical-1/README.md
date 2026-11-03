# Data Preprocessing & Feature Engineering: Customer Churn

Practical-1 project: preparing an e-commerce customer dataset for machine learning, with the goal of **predicting customer churn**.

> **Author:** `<YOUR NAME>`  |  **Course / Batch:** `<COURSE NAME>`

---

## 1. Objective

Act as a Junior Data Analyst: load customer data from multiple sources, understand and clean it, engineer features, explore it with EDA, and profile it so that it is ready for a churn prediction model.

**ML problem:** Binary classification. Predict whether a customer will churn (`Churn` = 1) or stay (`Churn` = 0) from purchase behaviour and profile data. Because only ~17% of customers churn, recall, F1-score and ROC-AUC are used instead of plain accuracy.

## 2. Dataset

| Item | Detail |
|---|---|
| Name | E-commerce Customer Churn dataset |
| Source | Kaggle: `<PASTE KAGGLE DATASET LINK HERE>` |
| Size | 5,630 customers, 20 columns |
| Target | `Churn` (0 = stays, 1 = churns) |
| Quality issues | ~4-5% missing values in 7 columns, inconsistent category labels, outliers, an ID column |

### Assumptions (please read)
- The dataset has **no `Age` or `Income` column**, but the task asks for them. Both were **simulated** with a fixed random seed (42), built from `CityTier` plus random noise, and kept **independent of churn**. Findings about them are **not real-world insights**.
- `Purchases` is represented by the real column `OrderCount`.
- The original CSV was split into **CSV, JSON and SQLite** parts to practise multi-source loading. The API part uses a public dummy REST API (`jsonplaceholder`); if there is no internet the notebook falls back to a small labelled offline sample.

## 3. Project Structure

```
.
├── README.md
├── requirements.txt
├── Data/
│   ├── Raw/          ecommerce_churn.csv (original), Customers_Part1.csv, Customers_Part2.json, Customers_Part3.db
│   ├── External/     API data
│   └── Processed/    cleaned_churn_data.csv, ml_ready_churn_data.csv
├── Notebooks/        Practical1_Data_Preprocessing.ipynb
├── Reports/          data_quality_report*.csv, raw_profiling_report.html, cleaned_profiling_report.html
├── Visualization/    saved charts (PNG)
├── Screenshots/      notebook / report screenshots
└── Theory/           Theory_Concepts.pdf
```

## 4. Steps Performed

| Part | Work done |
|---|---|
| **A. Fundamentals** | Notes on data analysis, data science project steps, ML problem statement, tensors with NumPy examples |
| **B. Data acquisition** | Loaded CSV (`read_csv`), JSON (`json.load` + `json_normalize`), SQL (`sqlite3` + `read_sql_query`), REST API (`requests`); merged with `concat` (5,630 x 20, verified against the original) |
| **C. Understanding & cleaning** | `head/info/describe`; missing values and duplicates report; merged inconsistent labels (`Phone`/`Mobile Phone`, `CC`/`Credit Card`, `COD`/`Cash on Delivery`, `Mobile`/`Mobile Phone`); median imputation; type correction; dropped `CustomerID`; IQR outlier check and capping |
| **Feature engineering** | Simulated `Age`/`Income`; created `AgeGroup`, `CashbackPerOrder`, `IsNewCustomer`; one-hot encoding; standardisation; stratified train/test split |
| **D. EDA** | Univariate (Age, Income, Purchases, Tenure, Cashback, target), bivariate (Gender vs Purchases, Income vs Churn, churn drivers), multivariate (correlation heatmap, pair plot) |
| **E. Profiling** | `ydata-profiling` reports before and after cleaning |

## 5. Key Insights

1. About **17%** of customers churn (imbalanced target).
2. **Tenure** is the strongest signal: churned customers average ~4 months vs ~11 months for retained customers.
3. Customers who **complained** churn at 31.7% vs 10.9% for those who did not.
4. **Single** customers (26.7%) and **Cash-on-Delivery / E-wallet** users (24.9% / 22.8%) churn more.
5. Lower **cashback** is associated with higher churn.
6. Gender has no practically meaningful effect on purchase count; simulated Age and Income show no relation with churn (as expected).

**Limitations:** relationships are associations, not causes; the satisfaction-score and days-since-last-order patterns look counter-intuitive and need domain validation; the API data is a dummy source not linked to customers.

## 6. How to Run

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook Notebooks/Practical1_Data_Preprocessing.ipynb
```
Run all cells from top to bottom. The notebook regenerates the split files, processed datasets, charts and profiling reports. Open the `.html` reports in a browser.

## 7. Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, SQLite, Requests, ydata-profiling, Jupyter Notebook.

## 8. Theory

Definitions of the concepts used (data analysis, project lifecycle, tensors, cleaning, feature engineering, EDA, profiling) are in [`Theory/Theory_Concepts.pdf`](Theory/Theory_Concepts.pdf).
