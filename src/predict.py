import pandas as pd
import joblib
import os
import shap
import numpy as np

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "churn_pipeline.pkl")
THRESH_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_threshold.pkl")

pipeline       = joblib.load(MODEL_PATH)
best_threshold = joblib.load(THRESH_PATH)


def predict_churn(user_input: dict) -> dict:
    """
    Predicts churn probability for a single customer.

    Reasoning:
    - Feature engineering must exactly match train.py
    - shap.TreeExplainer used instead of shap.Explainer because:
        shap.Explainer with a single-row background returns all zeros.
        TreeExplainer works directly on the XGBoost booster internals
        and returns exact Shapley values without needing background data.
    - Risk bands: <35% Low, 35-60% Medium, >60% High
    - Threshold from PR Curve (not hardcoded 0.5)
    """

    tenure          = int(user_input["tenure"])
    total_charges   = float(user_input["TotalCharges"])
    monthly_charges = float(user_input["MonthlyCharges"])

    # Feature engineering — must match train.py exactly
    avg_monthly_spend = total_charges / (tenure + 1)

    if tenure <= 12:   tenure_group = "New"
    elif tenure <= 24: tenure_group = "Growing"
    elif tenure <= 48: tenure_group = "Mature"
    else:              tenure_group = "Loyal"

    df = pd.DataFrame([{
        "Tenure Months":     tenure,
        "Monthly Charges":   monthly_charges,
        "Total Charges":     total_charges,
        "Avg Monthly Spend": avg_monthly_spend,
        "Contract":          str(user_input["Contract"]),
        "Internet Service":  str(user_input["InternetService"]),
        "Payment Method":    str(user_input["PaymentMethod"]),
        "Tenure Group":      tenure_group,
        "Senior Citizen":    str(user_input["SeniorCitizen"]),
        "Dependents":        str(user_input["Dependents"]),
        "Tech Support":      str(user_input["TechSupport"]),
        "Online Security":   str(user_input["OnlineSecurity"]),
    }])

    # Prediction
    prob       = pipeline.predict_proba(df)[0][1]
    prediction = int(prob >= best_threshold)

    # Risk bands
    if prob < 0.35:   risk_level = "Low"
    elif prob < 0.60: risk_level = "Medium"
    else:             risk_level = "High"

    # =========================
    # SHAP EXPLANATION
    # TreeExplainer works directly on XGBoost booster —
    # gives exact SHAP values without background data.
    # shap.Explainer (generic) returns zeros when only
    # a single row is passed as background.
    # =========================
    shap_explanation = []
    try:
        preprocessor  = pipeline.named_steps["preprocessor"]
        classifier    = pipeline.named_steps["classifier"]
        X_transformed = preprocessor.transform(df)
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

        explainer   = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)

        vals  = np.array(shap_values)[0]
        pairs = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)[:5]
        shap_explanation = [
            {"feature": f, "shap_value": round(float(v), 4)} for f, v in pairs
        ]
    except Exception:
        shap_explanation = []

    return {
        "churn_probability": round(float(prob), 4),
        "churn_prediction":  prediction,
        "risk_level":        risk_level,
        "threshold_used":    round(float(best_threshold), 4),
        "shap_explanation":  shap_explanation,
    }


if __name__ == "__main__":
    sample = {
        "tenure": 2, "MonthlyCharges": 85.0, "TotalCharges": 170.0,
        "Contract": "Month-to-month", "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check", "SeniorCitizen": "Yes",
        "Dependents": "No", "TechSupport": "No", "OnlineSecurity": "No",
    }
    result = predict_churn(sample)
    print(f"\n🔍 Prediction Result:")
    print(f"  Churn Probability : {result['churn_probability']:.2%}")
    print(f"  Prediction        : {'⚠️  Will Churn' if result['churn_prediction'] else '✅ Will Stay'}")
    print(f"  Risk Level        : {result['risk_level']}")
    print(f"  Threshold Used    : {result['threshold_used']}")
    print(f"\n🧠 SHAP — Top Factors:")
    for s in result["shap_explanation"]:
        arrow = "▲ increases" if s["shap_value"] > 0 else "▼ decreases"
        print(f"  {s['feature']:<35} {arrow} churn risk  ({s['shap_value']:+.4f})")