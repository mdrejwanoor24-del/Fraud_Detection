# ==============================
# 1. IMPORTS
# ==============================
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, recall_score, precision_score, f1_score
)

from imblearn.over_sampling import SMOTE

# Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ==============================
# 2. LOAD DATA
# ==============================
df = pd.read_csv(r"C:\Users\DELL\OneDrive\Documents\Fraud_Detection_Project\Data\cleaned_data.csv")

# ==============================
# 3. PREPROCESSING
# ==============================
# Drop ID columns
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True, errors='ignore')

# Target to int
df['isFraud'] = df['isFraud'].astype(int)

# Feature Engineering
df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
df['hour'] = df['step'] % 24

# Encode categorical (handles both 'object' and 'str' dtypes in newer pandas)
if not pd.api.types.is_numeric_dtype(df['type']):
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])

# ==============================
# 4. SPLIT (STRATIFIED)
# ==============================
X = df.drop('isFraud', axis=1)
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==============================
# 5. HANDLE IMBALANCE
# ==============================
# (A) Compute scale_pos_weight for XGBoost
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# (B) SMOTE (on train only)
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:", np.bincount(y_train_sm))

# ==============================
# 6. DEFINE MODELS + PARAMS
# ==============================

# --- XGBoost ---
xgb = XGBClassifier(
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    n_jobs=-1
)

xgb_params = {
    'n_estimators': [200, 400],
    'max_depth': [4, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

# --- LightGBM ---
lgb = LGBMClassifier(
    class_weight='balanced',
    n_jobs=-1
)

lgb_params = {
    'n_estimators': [200, 400],
    'max_depth': [5, 10, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 70]
}

# --- CatBoost ---
cat = CatBoostClassifier(
    verbose=0,
    auto_class_weights='Balanced'  # Use CatBoost's built-in handling
)

cat_params = {
    'iterations': [200, 400],
    'depth': [4, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1]
}

# ==============================
# 7. TUNING FUNCTION
# ==============================
def tune_model(model, params, X_train, y_train, name):
    print(f"\n=== Tuning {name} ===")

    search = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=10,
        scoring='roc_auc',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print(f"{name} Best Params:", search.best_params_)
    print(f"{name} Best CV AUC:", search.best_score_)

    return search.best_estimator_

# ==============================
# 8. TRAIN + COMPARE
# ==============================
models = {}

models['XGBoost'] = tune_model(xgb, xgb_params, X_train_sm, y_train_sm, "XGBoost")
models['LightGBM'] = tune_model(lgb, lgb_params, X_train_sm, y_train_sm, "LightGBM")
models['CatBoost'] = tune_model(cat, cat_params, X_train_sm, y_train_sm, "CatBoost")

# ==============================
# 9. EVALUATE (AUC)
# ==============================
results = {}

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    results[name] = auc
    print(f"\n{name} AUC: {auc:.4f}")

# Select best
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")

# ==============================
# 10. THRESHOLD OPTIMIZATION
# ==============================
y_prob = best_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.05)
best_thresh = 0.5
best_recall = 0

print("\nThreshold tuning:")

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)

    recall = recall_score(y_test, y_pred_t)
    precision = precision_score(y_test, y_pred_t)
    f1 = f1_score(y_test, y_pred_t)

    print(f"T={t:.2f} | Recall={recall:.3f} | Precision={precision:.3f} | F1={f1:.3f}")

    if recall > best_recall:
        best_recall = recall
        best_thresh = t

print("\nBest Threshold:", best_thresh)

# ==============================
# 11. FINAL EVALUATION
# ==============================
y_pred_final = (y_prob >= best_thresh).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))

print("Final AUC:", roc_auc_score(y_test, y_prob))

# ==============================
# 12. SAVE MODEL + METADATA
# ==============================
joblib.dump({
    "model": best_model,
    "threshold": best_thresh,
    "model_name": best_model_name,
    "features": X.columns.tolist()
}, "best_model.pkl")

print("\nModel saved as best_model.pkl")

