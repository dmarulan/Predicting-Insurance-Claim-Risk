# Predicting-Insurance-Claim-Risk

This project aims to build a machine learning model that predicts the probability of a driver initiating an insurance claim. The dataset used is the [Porto Seguro Safe Driver Prediction Dataset](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/data).

## 📌 Objective
Develop a robust predictive model to identify high-risk drivers and help insurance companies like Allstate manage risk, reduce fraud, and improve pricing strategies.

## 📊 Dataset Overview
- **Target**: Binary indicator of whether a claim was made.
- **Features**: Anonymized categorical, binary, and continuous variables.
- **Training Size**: 595,212 rows, 57 columns
- **Testing Size**: 892,816 rows, 57 columns

- The dataset is also very imbalanced. Class distribution: {0: 458814, 1: 17355}

## Load the dataset (via Google Colab)

Due to size constraints, the dataset is not stored in this GitHub repo.  
You can:

- Manually upload `train.csv` and `test.csv` to your Colab environment, **or**
- Mount your Google Drive and store files in `/content/drive/MyDrive/...`.

## 🔍 Project Steps
1. **Exploratory Data Analysis (EDA)**
2. **Data Cleaning & Imputation**
3. **Feature Engineering**
4. **Model Training & Evaluation** (XGBoost)
5. **Interpretability with SHAP**
6. **(Optional)** Deployment via Streamlit

## 📈 Evaluation Metrics
- AUC-ROC
- F1 Score
- F2 Score
- Precision/Recall

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

├── models/ # Saved model
├── notebooks/         # EDA & model development
├── src/ # Core modular ML code
│ ├── preprocess.py
│ ├── model_training.py
│ ├── utils.py
│ └── inference.py
├── main.py/           
├── requirements.txt   # Project library dependencies
└── README.md
```

---

**Author:** Daniel E. Marulanda  
Ph.D. Candidate – Industrial & Systems Engineering
Machine Learning Engineer | Data Scientist
Bilingual: English 🇺🇸 & Spanish
