# 🏦 Home Credit Default Risk Prediction 🚀
**Kaggle Competition - Top Tier Performance**

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange.svg)
![AUC](https://img.shields.io/badge/Score-0.78270-green.svg)

## 📋 Project Overview
مشروع Machine Learning متكامل للتنبؤ باحتمالية تعثر العملاء في سداد القروض. التحدي يكمن في التعامل مع بيانات ضخمة موزعة على 7 جداول مختلفة مع وجود نسبة كبيرة من القيم المفقودة (Missing Values).

تم الوصول إلى سكور **0.78270 (ROC-AUC)** باستخدام تقنيات متقدمة في هندسة الميزات (Feature Engineering) وضبط الخوارزميات (Hyperparameter Tuning).

## 🧠 Data Engineering Pipeline
المشروع بيقوم بعملية معالجة شاملة (End-to-End Pipeline) بتشمل:
1. **Data Aggregation**: تجميع البيانات من جداول (Bureau, POS_CASH, Installments, Credit Card) وربطها بالعميل الأساسي.
2. **Domain Features**: بناء ميزات مالية ذكية بناءً على خبرة المجال (Financial Ratios).
3. **Handling Missing Data**: التعامل الذكي مع القيم المفقودة باستخدام ميزات LightGBM المدمجة.
4. **Categorical Encoding**: تحويل البيانات النصية إلى `Category` لتحسين أداء الموديل.



## 🛠️ Feature Engineering (The Secret Sauce 🧪)
القفزة في السكور من 0.75 لـ 0.78 جاءت بسبب إضافة ميزات مالية مثل:
* **CREDIT_INCOME_RATIO**: حجم القرض بالنسبة لدخل العميل.
* **ANNUITY_INCOME_RATIO**: قدرة العميل على تحمل القسط الشهري من دخله.
* **EMPLOYED_AGE_RATIO**: نسبة سنوات العمل لسن العميل (مؤشر الاستقرار).
* **CREDIT_TERM**: مدة القرض التقريبية بناءً على القسط.

## 🤖 Modeling
استخدمنا وحش البيانات المجدولة **LightGBM** مع الضبط التالي لتحقيق أفضل توازن بين السرعة والدقة:
- **N-Estimators**: 5000 (With Early Stopping)
- **Learning Rate**: 0.01
- **Regularization**: Lambda L1/L2 applied to prevent overfitting.
- **Handling Imbalance**: `class_weight='balanced'` للتعامل مع قلة حالات التعثر مقارنة بالحالات العادية.



## 📁 Project Structure
```text
home-credit-ML/
├── data/               # Raw and processed datasets (Ignored by Git)
├── src/
│   └── main.py         # Full Pipeline (Loading -> Engineering -> Training)
├── notebook/           # Exploratory Data Analysis (EDA)
├── .gitignore          # Prevents uploading large data files
└── README.md           # Documentation
