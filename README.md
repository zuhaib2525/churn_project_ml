# 🔮 ChurnLens — Customer Churn Prediction

A machine learning web app that predicts customer churn using XGBoost with SHAP explainability.

## 📊 Model Performance
- **Best Model:** XGBoost (GridSearchCV Tuned)
- **ROC-AUC:** 0.8603 (5-Fold CV)
- **F1 Score:** 0.6409
- **Threshold:** 0.60 (PR Curve optimised)

## 🧠 Features
- 6-model comparison (LR, DT, RF, KNN, SVM, XGBoost)
- GridSearchCV hyperparameter tuning (108 configs)
- Class imbalance handled via scale_pos_weight
- PR Curve optimal threshold selection
- SHAP explainability for every prediction
- Streamlit dashboard with retention recommendations

## 🚀 Run Locally
```bash
pip install -r requirements.txt
python3 src/train.py
streamlit run app/app.py
```

## 📁 Project Structure
