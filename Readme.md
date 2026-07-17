# Home Credit Default Risk Risk Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Framework-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This repository contains a production-grade, end-to-end Machine Learning classification pipeline built to solve the **Kaggle Home Credit Default Risk** challenge. The primary objective is to estimate a client's credit risk profile and predict their probability of loan default using a massive relational dataset spread across multiple tables.

By successfully integrating data from 6 relational tables, engineering advanced historical risk metrics, and optimizing gradient boosted trees, this pipeline delivers a highly competitive performance.

---

## 📊 Model Performance

| Model | Validation Accuracy | Validation ROC-AUC | Recall (Class 1 - Defaults) |
| :--- | :---: | :---: | :---: |
| **Baseline Random Forest** | 76.21% | 0.7344 |
| **Optimized XGBoost Classifier** | **81.53%** | **0.7827** |

### 📈 Key Insights & Evaluation
* **Class Imbalance Resolution:** The dataset suffers from severe class imbalance (~8% default rate)[cite: 1]. Utilizing `scale_pos_weight` inside XGBoost significantly optimized the minority class classification, raising the Recall from **0.54** to **0.65** without collapsing the model's precision.
* **Metric of Success:** The systematic hyperparameter tuning and custom feature engineering pushed the ROC-AUC score to a competitive **0.7827**, placing this solution near the top leaderboard ranges.

---

## 🛠️ Setup & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/ahmedsherif999/house-credit-ML](https://github.com/ahmedsherif999/house-credit-ML)
cd house-credit-ML
