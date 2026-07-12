"""
train_model.py
---------------
Full training pipeline for Endometriosis Detection.

Flow:
1. Load data
2. Train/test split (stratified, to preserve class ratio)
3. Scale features (fit on train only -> avoid data leakage)
4. Handle class imbalance (oversample minority class in TRAINING data only)
5. Train 5 classification algorithms
6. Hyperparameter-tune each with GridSearchCV + k-fold Cross-Validation
7. Evaluate all tuned models on the untouched test set
8. Select the best model by Recall (medical context: false negatives are costly)
9. Save the best model + scaler + feature list for the Streamlit app
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.utils import resample

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("endometriosis_data.csv")
FEATURES = [c for c in df.columns if c != "diagnosis"]
X = df[FEATURES]
y = df["diagnosis"]

# ---------------------------------------------------------------------------
# 2. Train/test split (stratified so both sets keep the same class ratio)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 3. Scale features - fit ONLY on training data, then transform both
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Handle class imbalance - random oversample the minority class in TRAIN only
#    (Swap this block for imblearn's SMOTE if the package is available:
#     from imblearn.over_sampling import SMOTE
#     X_train_scaled, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_scaled, y_train))
# ---------------------------------------------------------------------------
train_df = pd.DataFrame(X_train_scaled, columns=FEATURES)
train_df["diagnosis"] = y_train.values

majority = train_df[train_df.diagnosis == 0]
minority = train_df[train_df.diagnosis == 1]
minority_upsampled = resample(
    minority, replace=True, n_samples=len(majority), random_state=RANDOM_STATE
)
train_balanced = pd.concat([majority, minority_upsampled])
X_train_bal = train_balanced[FEATURES].values
y_train_bal = train_balanced["diagnosis"].values

# ---------------------------------------------------------------------------
# 5 & 6. Define models + hyperparameter grids, tune with GridSearchCV + CV
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

model_grids = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"max_depth": [3, 5, 7, None], "min_samples_split": [2, 5, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "max_depth": [5, 10, None], "max_features": ["sqrt", "log2"]},
    ),
    "SVM": (
        SVC(probability=True, random_state=RANDOM_STATE),
        {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
    ),
}

results = {}
fitted_models = {}

for name, (estimator, param_grid) in model_grids.items():
    print(f"\nTuning {name} ...")
    grid = GridSearchCV(
        estimator, param_grid, cv=cv, scoring="recall", n_jobs=-1
    )
    grid.fit(X_train_bal, y_train_bal)
    best_model = grid.best_estimator_
    fitted_models[name] = best_model

    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    results[name] = {
        "best_params": grid.best_params_,
        "cv_best_recall": round(grid.best_score_, 4),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_precision": round(precision_score(y_test, y_pred), 4),
        "test_recall": round(recall_score(y_test, y_pred), 4),
        "test_f1": round(f1_score(y_test, y_pred), 4),
        "test_roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"  Best params: {grid.best_params_}")
    print(f"  Test -> Acc: {results[name]['test_accuracy']}  "
          f"Recall: {results[name]['test_recall']}  F1: {results[name]['test_f1']}  "
          f"ROC-AUC: {results[name]['test_roc_auc']}")

# ---------------------------------------------------------------------------
# 8. Select best model by Recall (then F1 as tiebreaker)
#    Rationale: in a diagnostic screening tool, missing a true case (false
#    negative) is costlier than flagging a healthy patient for extra review.
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda k: (results[k]["test_recall"], results[k]["test_f1"]))
best_model = fitted_models[best_name]

print(f"\n=== BEST MODEL: {best_name} ===")
print(json.dumps(results[best_name], indent=2))

# ---------------------------------------------------------------------------
# 9. Save model, scaler, feature order, and full comparison table
# ---------------------------------------------------------------------------
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

with open("feature_list.json", "w") as f:
    json.dump(FEATURES, f)

with open("model_comparison.json", "w") as f:
    json.dump({"best_model": best_name, "results": results}, f, indent=2)

print("\nSaved: best_model.pkl, scaler.pkl, feature_list.json, model_comparison.json")
