# Customer Churn Prediction — Data Preprocessing & EDA

A Junior Data Analyst project simulating a real-world workflow: taking raw, messy
customer purchase-behavior data and preparing it for a machine learning task —
predicting customer churn.

---

## Problem Statement

A consumer insights company has collected customer purchase behavior data.
The goal is to frame the churn-prediction problem as a machine learning task,
then clean, explore, and profile the dataset so it is ML-ready.

---

## Dataset

- Rows: 1,020 (before cleaning) → 1,000 (after removing duplicates)
- Columns: 18
- Target variable: Churn (Yes / No)

| Column | Description |
|---|---|
| customerID | Unique customer identifier |
| gender, SeniorCitizen, Partner, Dependents | Demographics |
| tenure | Months as a customer |
| PhoneService, InternetService, Contract, PaperlessBilling, PaymentMethod | Service & billing details |
| Age, Income, Purchases | Customer profile & purchase behavior |
| MonthlyCharges, TotalCharges | Billing amounts |
| Churn | Target — whether the customer churned |

The raw dataset was intentionally messy — missing values, duplicate rows,
inconsistent data types, and an irrelevant column — to mirror real-world
data quality issues.

---

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

---

## Project Workflow

### 1. Data Loading
Loaded the raw CSV into a Pandas DataFrame and inspected its shape and columns.

<img width="875" height="608" alt="image" src="https://github.com/user-attachments/assets/24047242-d6ef-4fe0-83ce-9672708c1989" />


### 2. Data Understanding
Used .head() and .info() to inspect structure, data types, and non-null counts
across all 18 columns.

<img width="873" height="701" alt="image" src="https://github.com/user-attachments/assets/11eefe31-8e32-4343-8a8f-cf394a374cf4" />



### 3. Missing Values & Duplicates
Identified missing values across several columns (Dependents, InternetService,
Age, Income, TotalCharges) and 20 duplicate rows, then removed the duplicates.

<img width="872" height="623" alt="image" src="https://github.com/user-attachments/assets/e24bdb51-1fd2-47d6-b33f-592c70563971" />


### 4. Data Cleaning
- Fixed inconsistent data types (e.g. Age stored as "45 yrs", TotalCharges
  stored as " 2158.73 ") by extracting numeric values and converting columns
  to the correct dtype.
- Imputed missing values (median for numeric columns).
- Dropped the irrelevant Notes column.

<img width="1014" height="579" alt="image" src="https://github.com/user-attachments/assets/cc17d6b6-34ec-49ca-abee-75a16131468c" />



### 5. Univariate Analysis
Explored the distribution of Age, Income, and Purchases using histograms
with KDE overlays.

<img width="872" height="483" alt="image" src="https://github.com/user-attachments/assets/4371d241-ee57-441a-974c-7b858239636a" />


### 6. Bivariate Analysis
Compared Purchases across gender, and Income across Churn status using
boxplots.

<img width="1544" height="573" alt="image" src="https://github.com/user-attachments/assets/b4ae8a83-6b26-40c8-a01e-7fc416dc184c" />


### 7. Multivariate Analysis
Generated a correlation heatmap across all numeric features to understand
feature relationships — e.g. tenure and TotalCharges show strong positive
correlation (0.75).

<img width="874" height="656" alt="image" src="https://github.com/user-attachments/assets/5d52e2c6-ad06-4bdb-b821-48f400695225" />


---

## Key Insights

- tenure and TotalCharges are strongly correlated (0.75), as expected —
  longer-tenured customers accumulate higher total charges.
- Purchases correlates moderately with both tenure (0.62) and MonthlyCharges (0.46).
- Age and Income show little correlation with other numeric features,
  suggesting they act as independent signals for the churn model.
- Income distributions across churned vs. retained customers are broadly similar,
  suggesting churn is driven more by contract/service factors than income alone.

---

## Machine Learning Framing

- Type: Supervised, Binary Classification
- Input (X): Customer demographics, service usage, and billing attributes
- Output (y): Churn — Yes / No
- Goal: Predict at-risk customers in advance so the business can act to retain them

---

## Project Structure

├── customer_churn_dataset.csv     # Raw dataset
├── DataProfiler.ipynb             # Main notebook (cleaning + EDA)
├── screenshots/                   # Notebook output screenshots
└── README.md

---
