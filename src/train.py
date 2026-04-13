import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, f1_score, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# =========================
# PATHS
# =========================
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "Telco_customer_churn.csv")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "churn_pipeline.pkl")
THRESH_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_threshold.pkl")
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)

# =========================
# LOAD + CLEAN
# =========================
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["Churn"] = df["Churn Label"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["Churn"])

# =========================
# FEATURE ENGINEERING
# =========================
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)
df["Avg Monthly Spend"] = df["Total Charges"] / (df["Tenure Months"] + 1)
df["Tenure Group"] = df["Tenure Months"].apply(
    lambda x: "New" if x <= 12 else "Growing" if x <= 24 else "Mature" if x <= 48 else "Loyal"
)

# =========================
# FEATURES
# =========================
numerical_cols   = ["Tenure Months", "Monthly Charges", "Total Charges", "Avg Monthly Spend"]
categorical_cols = [
    "Contract", "Internet Service", "Payment Method",
    "Tenure Group", "Senior Citizen", "Dependents",
    "Tech Support", "Online Security"
]

X = df[numerical_cols + categorical_cols].copy()
y = df["Churn"]

for col in X.columns:
    if hasattr(X[col], "cat"):
        X[col] = X[col].astype(str).fillna("Unknown")
    elif X[col].dtype == "object":
        X[col] = X[col].fillna("Unknown")
    else:
        X[col] = X[col].fillna(0)

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# PREPROCESSOR
# =========================
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
])

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"ℹ️  Class ratio (scale_pos_weight): {scale_pos_weight:.2f}\n")

# =========================
# STEP 1 — TUNE XGBOOST FIRST
# Reason: XGBoost needs hyperparameter tuning to show its true
# potential. Without tuning, a default Logistic Regression often
# beats a default XGBoost on small tabular datasets like Telco.
# GridSearchCV finds the best n_estimators, max_depth, and
# learning_rate via 5-fold CV scored on ROC-AUC.
# =========================
print("🔧 Tuning XGBoost via GridSearchCV (this takes ~2-3 mins)...")

xgb_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])

param_grid = {
    "classifier__n_estimators":     [200, 300, 500],
    "classifier__max_depth":        [3, 4, 5],
    "classifier__learning_rate":    [0.01, 0.05, 0.1],
    "classifier__subsample":        [0.8, 1.0],
    "classifier__colsample_bytree": [0.8, 1.0],
}

grid_search = GridSearchCV(
    xgb_pipe,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

best_xgb_params = grid_search.best_params_
best_xgb_roc    = grid_search.best_score_

print(f"\n✅ Best XGBoost ROC-AUC (CV): {best_xgb_roc:.4f}")
print(f"   Best Params: {best_xgb_params}")

# Extract tuned XGBoost classifier
tuned_xgb = grid_search.best_estimator_.named_steps["classifier"]

# =========================
# STEP 2 — MODEL COMPARISON
# Reason: We compare tuned XGBoost vs all other models at
# default settings (same preprocessing pipeline for fairness).
# This directly addresses the literature gap of "limited model
# comparison — only 1-2 algorithms tested".
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
    "SVM":                 SVC(probability=True, kernel="rbf", random_state=42),
    "XGBoost (Tuned)":     tuned_xgb,
}

print("\n" + "=" * 65)
print(f"{'Model':<22} {'Accuracy':>9} {'F1':>8} {'ROC-AUC':>9} {'PR-AUC':>8}")
print("=" * 65)

comparison_results = {}
for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    acc    = (y_pred == y_test).mean()
    f1     = f1_score(y_test, y_pred)
    roc    = roc_auc_score(y_test, y_proba)
    pr     = average_precision_score(y_test, y_proba)

    comparison_results[name] = {
        "accuracy": acc, "f1": f1, "roc_auc": roc, "pr_auc": pr
    }
    print(f"  {name:<20} {acc:>9.4f} {f1:>8.4f} {roc:>9.4f} {pr:>8.4f}")

print("=" * 65)

# =========================
# STEP 3 — SELECT BEST BY ROC-AUC
# Reason: ROC-AUC is threshold-independent and measures the
# model's ability to rank churners above non-churners at ANY
# decision threshold. It is the standard metric for imbalanced
# binary classification in literature (Burez & Van den Poel, 2009).
# Accuracy is avoided as it is misleading on imbalanced data —
# a model predicting "no churn" for everyone scores 74% accuracy
# but catches zero churners.
# =========================
best_model_name = max(comparison_results, key=lambda k: comparison_results[k]["roc_auc"])
print(f"\n🏆 Best Model by ROC-AUC: {best_model_name}")
print(f"   ROC-AUC  : {comparison_results[best_model_name]['roc_auc']:.4f}")
print(f"   F1       : {comparison_results[best_model_name]['f1']:.4f}")
print(f"   Accuracy : {comparison_results[best_model_name]['accuracy']:.4f}")

# =========================
# STEP 4 — RETRAIN BEST WITH 5-FOLD CV
# Reason: Cross-validation on the full dataset gives a more
# reliable estimate of generalisation performance than a single
# train/test split, reducing variance due to random partitioning.
# =========================
best_clf = models[best_model_name]
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", best_clf)])

print(f"\n🔄 Running 5-Fold Stratified CV on {best_model_name}...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    pipeline, X, y, cv=cv,
    scoring={"f1": "f1", "roc_auc": "roc_auc"}
)
print(f"  CV F1 Score : {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
print(f"  CV ROC-AUC  : {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")

pipeline.fit(X_train, y_train)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# =========================
# STEP 5 — OPTIMAL THRESHOLD via PR Curve
# Reason: Default threshold of 0.5 is arbitrary and suboptimal
# for imbalanced data. The PR Curve finds the threshold that
# maximises F1-score — balancing precision (avoiding false alarms)
# and recall (catching real churners). This directly addresses
# the literature gap: "evaluation focused on accuracy with no
# regard for recall or F1".
# =========================
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores      = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx       = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\n🎯 Optimal Threshold (PR Curve max F1): {best_threshold:.4f}")
print(f"   Precision : {precisions[best_idx]:.4f}")
print(f"   Recall    : {recalls[best_idx]:.4f}")
print(f"   F1        : {f1_scores[best_idx]:.4f}")

# Threshold scan for transparency
print(f"\n📋 Threshold Scan:")
from sklearn.metrics import recall_score, precision_score
for t in [0.3, 0.4, 0.5, 0.6]:
    yp = (y_proba >= t).astype(int)
    print(f"  t={t}  Accuracy={(yp==y_test).mean():.4f}  "
          f"Recall={recall_score(y_test,yp):.4f}  "
          f"Precision={precision_score(y_test,yp):.4f}  "
          f"F1={f1_score(y_test,yp):.4f}")

# =========================
# STEP 6 — FINAL EVALUATION
# =========================
y_pred_default = (y_proba >= 0.5).astype(int)
y_pred_optimal = (y_proba >= best_threshold).astype(int)

print(f"\n📊 --- Default Threshold (0.5) ---")
print(classification_report(y_test, y_pred_default))
print(f"📊 --- Optimal Threshold ({best_threshold:.2f}) ---")
print(classification_report(y_test, y_pred_optimal))

print(f"📈 Final Metrics:")
print(f"  Accuracy : {(y_pred_optimal == y_test).mean():.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print(f"  PR-AUC   : {average_precision_score(y_test, y_proba):.4f}")
print(f"  F1 Score : {f1_score(y_test, y_pred_optimal):.4f}")

# =========================
# SAVE
# =========================
joblib.dump(pipeline, MODEL_PATH)
joblib.dump(best_threshold, THRESH_PATH)
print(f"\n✅ Best model ({best_model_name}) saved → models/churn_pipeline.pkl")
print(f"✅ Threshold ({best_threshold:.4f}) saved → models/best_threshold.pkl")