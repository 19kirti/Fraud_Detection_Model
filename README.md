# 🕵️‍♂️ RiskRadar: Fraud Detection Model

A machine learning project for detecting fraudulent transactions using multiple classification algorithms — **Random Forest**, **XGBoost**, and **Logistic Regression**.  
The goal is to identify fraudulent activity with high accuracy and minimal false positives.

---

## 🚀 Project Overview

Fraud detection is a critical challenge in financial systems where fraudulent transactions are rare but costly.  
This project explores different machine learning approaches to build a robust fraud detection pipeline.

The models were trained and evaluated on a labeled dataset (fraudulent vs. legitimate transactions) using key performance metrics like precision, recall, F1-score, and ROC-AUC.

---

## 🧠 Models Used

| Model | Accuracy | Key Notes |
|--------|-----------|-----------|
| **Random Forest Classifier** | ⭐ Best performing | Handles imbalance and non-linearity well |
| **XGBoost Classifier** | ⭐ Best performing | Excellent boosting performance, handles skewed data |
| **Logistic Regression** | Good baseline | Interpretable but limited for complex relationships |

---

## ⚙️ Tech Stack

- **Language:** Python 🐍  
- **Libraries:**  
  - `pandas`, `numpy` — data handling  
  - `matplotlib`, `seaborn` — visualization  
  - `scikit-learn` — Random Forest, Logistic Regression, metrics  
  - `xgboost` — gradient boosting  
  - `imbalanced-learn` — SMOTE or class balancing  
  - `jupyter notebook` — experimentation

---

## 📊 Workflow

1. **Data Preprocessing**
   - Handled missing values, scaled numerical features  
   - Encoded categorical variables  
   - Balanced data using SMOTE (if required)

2. **Exploratory Data Analysis (EDA)**
   - Visualized class imbalance, correlations, and feature distributions

3. **Model Training**
   - Split data into training and testing sets  
   - Trained multiple models and tuned hyperparameters

4. **Evaluation**
   - Compared models using metrics:
     - Precision
     - Recall
     - F1-score
     - ROC-AUC

5. **Result**
   - Random Forest and XGBoost achieved the best trade-off between recall and precision.

---

## 🧾 Results Summary

- **Random Forest ROC-AUC:** ~0.96  
- **XGBoost ROC-AUC:** ~0.97  
- **Logistic Regression ROC-AUC:** ~0.95

## 📈 Future Improvements

- Add deep learning model (e.g., LSTM or Autoencoder for anomaly detection)
- Use cross-validation for more robust metrics
- Deploy via Streamlit or Flask web app
- Integrate explainability tools (e.g., SHAP, LIME)

## 🏆 Acknowledgements
- Dataset inspired by public fraud detection datasets (e.g., Kaggle)
- Libraries: scikit-learn, XGBoost
