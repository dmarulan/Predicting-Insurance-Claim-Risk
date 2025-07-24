# Predicting-Insurance-Claim-Risk

This project aims to build a machine learning model that predicts the probability of a driver initiating an insurance claim. The dataset used is the [Porto Seguro Safe Driver Prediction Dataset](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/data).

## 📌 Objective
Develop a robust predictive model to identify high-risk drivers and help insurance companies like Allstate manage risk, reduce fraud, and improve pricing strategies.

## 📊 Dataset Overview
- **Target**: Binary indicator of whether a claim was made.
- **Features**: Anonymized categorical, binary, and continuous variables.
- **Size**: 595,212 rows, 57 columns

## 🔍 Project Steps
1. **Exploratory Data Analysis (EDA)**
2. **Data Cleaning & Imputation**
3. **Feature Engineering**
4. **Model Training & Evaluation** (Logistic Regression, XGBoost, LightGBM)
5. **Interpretability with SHAP**
6. **(Optional)** Deployment via Streamlit

## 📈 Evaluation Metrics
- AUC-ROC
- F1 Score
- Precision/Recall
- Log Loss

## 💡 Business Impact
- Improve claim forecasting accuracy
- Optimize risk-based premium pricing
- Enhance underwriting decisions

## 🛠 Tools Used
- Python, Scikit-learn, XGBoost, LightGBM
- SHAP for model interpretability
- Matplotlib & Seaborn for visualization

## 📂 Structure
```
insurance-risk-prediction/
├── data/              # Raw & processed data
├── notebooks/         # EDA & model development
├── src/               # Feature engineering & training scripts
├── models/            # Trained models
├── reports/           # Evaluation and plots
├── requirements.txt   # Project library dependencies
└── README.md
```

---

**Author:** Daniel E. Marulanda  
**Contact:** dem334@msstate.edu
'''
