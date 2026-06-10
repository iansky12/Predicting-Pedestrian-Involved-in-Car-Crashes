# Traffic Incident Analysis & Predictive Modeling

A data science project focused on cleaning, processing, and executing machine learning workflows on raw vehicle crash datasets. This repository contains modular workflows optimized for data preprocessing, stratified evaluation, and predictive modeling using tree-based ensemble algorithms.

## 🛠️ Tech Stack & Tools
* **Language:** R[cite: 1, 2]
* **Machine Learning Stack:** `xgboost`, `caret`, `ranger` (Random Forest)[cite: 1, 2]
* **Data Evaluation:** `pROC`, `yardstick`[cite: 1, 2]
* **Data Engineering & Viz:** `tidyverse`, `janitor`, `ggplot2`, `readr`, `forcats`[cite: 1, 2]

## 📊 Core Architecture & Pipelines

### 1. Robust Preprocessing Pipeline
* **Class Isolation:** Automatically cleans human-entered column tags, filters missing targets, and drops unrecognized factor classes to maintain binary target setups[cite: 1, 2].
* **Imputation Strategy:** Numerical columns are handled via median imputation, while character/categorical missing fields are safely imputed as "Unknown"[cite: 1, 2].
* **Dimensionality Reduction:** Features are streamlined by collapsing low-frequency categorical levels (`min_n = 200`) and dropping near-zero-variance (NZV) predictors via stratified feature variance filtering[cite: 1, 2].

### 2. Implemented Estimators

* **XGBoost (Extreme Gradient Boosting):** Trained using direct sparse matrix formatting (`xgb.DMatrix`), optimized with cross-validated area under the curve (`auc`) metrics, a logistics objective function, and parallelized execution[cite: 1].
* **Random Forest (Fast ranger execution):** Implemented in optimized execution mode to build cross-verified forest structures using Gini-split rule splits, parallelized core tracking, and automated baseline accuracy threshold comparisons[cite: 2].

## 💻 How to Run

### Prerequisites
Ensure you have an R environment configured. The script will automatically scan your local directory and install any missing packages from the CRAN repository[cite: 1, 2].

### Configuration
Open either script (`RGBOOST.Rmd` or `Memo_3.Rmd`) and update the local data layout configuration path pointing toward your system storage location[cite: 1, 2]:

```r
DATA_PATH <- "path/to/your/Crash_Data_Cleaned_v3.csv"
