# 💳 Fraud Detection System (End-to-End ML Project)

## 📌 Overview

This project builds a complete **fraud detection pipeline** using machine learning on transactional data.
It handles **imbalanced data**, compares multiple models, and provides **interpretable predictions** using SHAP.

---

## 🎯 Objective

Detect fraudulent transactions with high recall while maintaining good precision in a highly imbalanced dataset.

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost, LightGBM, CatBoost
* SHAP
* Streamlit

---

## 📊 Features

* step (time of transaction)
* type (transaction type)
* amount
* oldbalanceOrg, newbalanceOrig
* oldbalanceDest, newbalanceDest

### 🔧 Feature Engineering

* balance_diff_orig
* balance_diff_dest
* hour

---

## 🚀 Pipeline

1. Data Cleaning & Preprocessing
2. Feature Engineering
3. Handling Imbalance (SMOTE + class weights)
4. Model Training (XGBoost, LightGBM, CatBoost)
5. Hyperparameter Tuning (RandomizedSearchCV)
6. Model Evaluation (ROC-AUC, Recall, F1)
7. Threshold Optimization
8. Model Explainability (SHAP)
9. Deployment (Streamlit)

---

## 📈 Model Performance

* Best Model: LightGBM / XGBoost
* AUC: ~0.99
* High Recall for fraud detection

---

## 🧠 Explainability (SHAP)

* Transaction type is the strongest feature
* Balance differences heavily influence predictions
* High transaction amounts increase fraud probability

---

## 📁 Project Structure

```
Fraud_Detection_Project/
│
├── data/
│   ├── Fraud_Analysis_Dataset.csv
│   └── cleaned_data.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Preprocessing.ipynb
│   └── Shap.ipynb
│
├── src/
│   ├── training.py
│   └── predict.py
│
├── app.py
├── best_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ Setup & Run

### 1️⃣ Clone repo

```
git clone <your-repo-link>
cd Fraud_Detection_Project
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Train model

```
python src/training.py
```

### 4️⃣ Run app

```
streamlit run app.py
```

---

## 🌐 Deployment

You can deploy this project using:

* Streamlit Cloud
* Render
* Hugging Face Spaces

---

## 🎤 Key Highlights

* Solved imbalanced classification problem
* Used SMOTE + class weighting
* Compared 3 boosting models
* Applied threshold tuning for better fraud detection
* Explained predictions using SHAP

---

## 📌 Future Work

* Real-time API deployment
* Dashboard monitoring
* Deep learning models

---

## 🙌 Author

Your Name
MD REJWANOOR RAHAMAN