import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import predict_churn

st.set_page_config(
    page_title="ChurnLens",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 3rem !important; max-width: 1100px; }

    .hero { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 2rem; border: 1px solid rgba(79,152,163,0.3); }
    .hero-title { font-size: 2rem; font-weight: 700; color: #fff; letter-spacing: -0.02em; }
    .hero-sub { color: #4f98a3; font-size: 0.875rem; margin-top: 0.3rem; font-weight: 500; }
    .hero-badges { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .badge { background: rgba(79,152,163,0.15); border: 1px solid rgba(79,152,163,0.35); color: #4f98a3; border-radius: 99px; padding: 3px 12px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; }

    div[data-testid="stForm"] { background: #1a1917 !important; border: 1px solid #2c2b28 !important; border-radius: 16px !important; padding: 2rem !important; }
    .sec-label { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #4f98a3; margin-bottom: 0.8rem; padding-bottom: 0.5rem; border-bottom: 1px solid #2c2b28; }

    .stButton > button { background: linear-gradient(135deg, #01696f, #0c4e54) !important; color: #fff !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; font-size: 0.95rem !important; }
    .stButton > button:hover { box-shadow: 0 4px 20px rgba(1,105,111,0.45) !important; transform: translateY(-1px) !important; }

    .result-card { border-radius: 16px; padding: 1.8rem 2rem; margin: 1.5rem 0; border: 1.5px solid; }
    .result-card.high   { background: linear-gradient(135deg,#2a1518,#1a0d0f); border-color: #dd6974; }
    .result-card.medium { background: linear-gradient(135deg,#2d1e14,#1c1109); border-color: #bb653b; }
    .result-card.low    { background: linear-gradient(135deg,#1e2e17,#111d0e); border-color: #6daa45; }
    .result-icon  { font-size: 2.5rem; line-height: 1; margin-bottom: 0.5rem; }
    .result-risk  { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
    .result-risk.high   { color: #dd6974; }
    .result-risk.medium { color: #bb653b; }
    .result-risk.low    { color: #6daa45; }
    .result-title { font-size: 1.4rem; font-weight: 700; color: #fff; }
    .result-desc  { font-size: 0.875rem; color: #797876; margin-top: 0.4rem; }

    .metrics-row { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
    .m-tile { flex: 1; min-width: 130px; background: #1f1e1c; border: 1px solid #2c2b28; border-radius: 12px; padding: 1rem 1.2rem; text-align: center; }
    .m-label { font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #4a4947; margin-bottom: 0.35rem; }
    .m-value { font-size: 1.35rem; font-weight: 700; color: #cdccca; }

    .panel { background: #1a1917; border: 1px solid #2c2b28; border-radius: 12px; padding: 1.2rem; }
    .panel-title { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #4a4947; margin-bottom: 0.8rem; }
    .p-row { display: flex; align-items: center; gap: 0.7rem; padding: 0.45rem 0; border-bottom: 1px solid #1f1e1c; }
    .p-row:last-child { border-bottom: none; }
    .p-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .d-risk { background: #dd6974; } .d-safe { background: #6daa45; } .d-neutral { background: #4a4947; }
    .p-text { flex: 1; font-size: 0.875rem; color: #cdccca; }
    .p-tag  { font-size: 0.68rem; font-weight: 600; padding: 2px 9px; border-radius: 99px; }
    .t-risk    { background: rgba(221,105,116,0.15); color: #dd6974; }
    .t-safe    { background: rgba(109,170,69,0.15);  color: #6daa45; }
    .t-neutral { background: rgba(74,73,71,0.2);     color: #797876; }
</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">📊 ChurnLens</div>
    <div class="hero-sub">Customer Churn Prediction — XGBoost (GridSearchCV Tuned) · Telco Dataset</div>
    <div class="hero-badges">
        <span class="badge">XGBoost Tuned</span>
        <span class="badge">GridSearchCV</span>
        <span class="badge">ROC-AUC Selected</span>
        <span class="badge">PR Curve Threshold</span>
        <span class="badge">SHAP Explainability</span>
        <span class="badge">6-Model Comparison</span>
        <span class="badge">5-Fold CV</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FORM ─────────────────────────────────────────────────
with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="sec-label">💰 Financial Details</div>', unsafe_allow_html=True)
        tenure  = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
        total   = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0, step=1.0)

    with col2:
        st.markdown('<div class="sec-label">🔧 Service & Contract</div>', unsafe_allow_html=True)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment  = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    with col3:
        st.markdown('<div class="sec-label">👤 Customer Profile</div>', unsafe_allow_html=True)
        senior          = st.radio("Senior Citizen",  ["No", "Yes"], horizontal=True)
        dependents      = st.radio("Has Dependents",  ["No", "Yes"], horizontal=True)
        tech_support    = st.radio("Tech Support",    ["No", "Yes"], horizontal=True)
        online_security = st.radio("Online Security", ["No", "Yes"], horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("⚡ Predict Churn", use_container_width=True, type="primary")

# ── RESULT ───────────────────────────────────────────────
if submitted:
    with st.spinner("Running XGBoost + SHAP..."):
        try:
            result = predict_churn({
                "tenure":          tenure,
                "MonthlyCharges":  monthly,
                "TotalCharges":    total,
                "Contract":        contract,
                "InternetService": internet,
                "PaymentMethod":   payment,
                "SeniorCitizen":   senior,
                "Dependents":      dependents,
                "TechSupport":     tech_support,
                "OnlineSecurity":  online_security,
            })

            prob       = result["churn_probability"]
            prediction = result["churn_prediction"]
            risk       = result["risk_level"]
            threshold  = result["threshold_used"]
            shap_data  = result["shap_explanation"]
            avg_spend  = total / (tenure + 1)

            # ── Result Card
            cfg = {
                "High":   ("high",   "🚨", "High Risk",   "Immediate intervention recommended — trigger a retention campaign now."),
                "Medium": ("medium", "⚠️", "Medium Risk", "Monitor closely — consider a proactive retention offer."),
                "Low":    ("low",    "✅", "Low Risk",    "Strong retention signals — no immediate action needed."),
            }
            cls, icon, label, desc = cfg[risk]
            st.markdown(f"""
            <div class="result-card {cls}">
                <div class="result-icon">{icon}</div>
                <div class="result-risk {cls}">{label}</div>
                <div class="result-title">{'Customer likely to churn' if prediction else 'Customer likely to stay'}</div>
                <div class="result-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

            # ── Metrics
            prob_color = "#dd6974" if risk=="High" else "#bb653b" if risk=="Medium" else "#6daa45"
            tg = "New 📦" if tenure<=12 else "Growing 🌱" if tenure<=24 else "Mature 🏗️" if tenure<=48 else "Loyal ⭐"
            st.markdown(f"""
            <div class="metrics-row">
                <div class="m-tile"><div class="m-label">Churn Probability</div><div class="m-value" style="color:{prob_color}">{prob:.1%}</div></div>
                <div class="m-tile"><div class="m-label">Prediction</div><div class="m-value" style="font-size:0.9rem">{'Will Churn 🔴' if prediction else 'Will Stay 🟢'}</div></div>
                <div class="m-tile"><div class="m-label">Threshold</div><div class="m-value">{threshold:.2f}</div></div>
                <div class="m-tile"><div class="m-label">Avg Monthly Spend</div><div class="m-value">${avg_spend:.0f}</div></div>
                <div class="m-tile"><div class="m-label">Tenure Group</div><div class="m-value" style="font-size:0.85rem">{tg}</div></div>
            </div>""", unsafe_allow_html=True)

            st.progress(prob, text=f"Churn Risk Score: {prob:.1%}")

            # ── Three Panels
            c1, c2, c3 = st.columns(3)

            # ── Panel 1: Feature Signals
            with c1:
                signals = []
                if contract == "Month-to-month": signals.append(("risk", "Month-to-month contract", "Top churn driver"))
                elif contract == "Two year":     signals.append(("safe", "Two-year contract",        "Strong retention"))
                if internet == "Fiber optic":    signals.append(("risk", "Fiber optic service",      "High churn segment"))
                if payment == "Electronic check":signals.append(("risk", "Electronic check",         "Correlated with churn"))
                elif "automatic" in payment:     signals.append(("safe", "Automatic payment",        "Retention signal"))
                if tenure <= 6:                  signals.append(("risk", f"Only {tenure}mo tenure",  "New = at-risk"))
                elif tenure >= 48:               signals.append(("safe", f"{tenure}mo tenure",       "Loyal customer"))
                if online_security == "Yes":     signals.append(("safe", "Online Security active",   "Lowers churn risk"))
                if tech_support == "Yes":        signals.append(("safe", "Tech Support active",      "Lowers churn risk"))
                if dependents == "Yes":          signals.append(("safe", "Has dependents",           "Strong retention"))
                if senior == "Yes":              signals.append(("neutral", "Senior citizen",        "Demographic factor"))

                rows = "".join([
                    f'<div class="p-row"><div class="p-dot d-{t}"></div>'
                    f'<span class="p-text">{tx}</span>'
                    f'<span class="p-tag t-{t}">{tg2}</span></div>'
                    for t, tx, tg2 in signals
                ])
                st.markdown(
                    f'<div class="panel"><div class="panel-title">📌 Feature Signals</div>{rows}</div>',
                    unsafe_allow_html=True
                )

            # ── Panel 2: SHAP with human-readable reasoning
            with c2:
                reason_map = {
                    "Tenure Months":                           ("short tenure means less loyalty",                    "long tenure means strong loyalty"),
                    "Monthly Charges":                         ("high monthly cost increases churn risk",             "low monthly cost reduces churn risk"),
                    "Total Charges":                           ("high total spend indicates long-term customer",      "low total spend suggests at-risk customer"),
                    "Avg Monthly Spend":                       ("high avg spend increases risk",                     "low avg spend suggests value gap"),
                    "Contract_Month-to-month":                 ("month-to-month contract is top churn driver",       "absence of month-to-month reduces risk"),
                    "Contract_One year":                       ("one-year contract improves retention",              "no one-year contract slightly raises risk"),
                    "Contract_Two year":                       ("two-year contract strongly reduces churn",          "no long-term contract raises churn risk"),
                    "Internet Service_Fiber optic":            ("fiber optic users churn most frequently",           "non-fiber users are more stable"),
                    "Internet Service_DSL":                    ("DSL users have lower churn rate",                   "absence of DSL slightly raises risk"),
                    "Payment Method_Electronic check":         ("electronic check strongly correlated with churn",   "non-electronic check lowers risk"),
                    "Payment Method_Bank transfer (automatic)":("auto bank transfer indicates stability",            "no auto payment raises slight risk"),
                    "Payment Method_Credit card (automatic)":  ("auto credit card indicates stability",              "no auto payment raises slight risk"),
                    "Online Security_No":                      ("no online security increases churn risk",           "online security active reduces risk"),
                    "Online Security_Yes":                     ("online security active reduces churn risk",         "no security raises risk"),
                    "Tech Support_No":                         ("no tech support increases churn risk",              "tech support active reduces risk"),
                    "Tech Support_Yes":                        ("tech support active reduces churn risk",            "no support raises risk"),
                    "Dependents_No":                           ("no dependents means less retention anchor",         "has dependents strongly reduces churn"),
                    "Dependents_Yes":                          ("has dependents strongly reduces churn",             "no dependents raises risk"),
                    "Senior Citizen_Yes":                      ("senior citizen demographic factor",                 "non-senior demographic factor"),
                    "Tenure Group_New":                        ("new customer — highest churn risk group",           "not in new group — lower risk"),
                    "Tenure Group_Loyal":                      ("loyal customer — lowest churn risk group",          "not in loyal group"),
                }

                if shap_data:
                    shap_rows = ""
                    for s in shap_data:
                        v      = s["shap_value"]
                        feat   = s["feature"]
                        color  = "#dd6974" if v > 0 else "#6daa45"
                        arrow  = "▲" if v > 0 else "▼"
                        impact = "increases" if v > 0 else "reduces"
                        reason = reason_map.get(feat, (f"{feat.replace('_',' ')} {impact} churn risk",) * 2)
                        reason = reason[0] if v > 0 else reason[1]

                        shap_rows += f"""
                        <div style="display:flex;align-items:flex-start;gap:0.6rem;padding:0.55rem 0;border-bottom:1px solid #1f1e1c;">
                            <span style="color:{color};font-weight:700;font-size:1.1rem;margin-top:2px;width:16px">{arrow}</span>
                            <div style="flex:1">
                                <div style="font-size:0.82rem;font-weight:600;color:#cdccca">{feat.replace('_', ' ')}</div>
                                <div style="font-size:0.74rem;color:#797876;margin-top:2px">{reason}</div>
                            </div>
                            <span style="font-size:0.75rem;font-weight:600;color:{color};white-space:nowrap">{v:+.3f}</span>
                        </div>"""

                    st.markdown(
                        f'<div class="panel"><div class="panel-title">🧠 SHAP — Why This Prediction?</div>{shap_rows}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="panel"><div class="panel-title">🧠 SHAP</div>'
                        '<div style="color:#4a4947;font-size:0.85rem;padding:0.5rem 0">SHAP values not available.</div></div>',
                        unsafe_allow_html=True
                    )

            # ── Panel 3: Retention Actions
            with c3:
                retention = {
                    "High": [
                        ("🎯", "Contract Upgrade", "Offer 1-year plan at 20% discount"),
                        ("💰", "Billing Credit",   "Apply 15% credit for 3 months"),
                        ("📞", "Priority Call",    "Trigger customer success call today"),
                        ("🛡️", "Free Add-ons",    "Free Tech Support + Security 60 days"),
                    ],
                    "Medium": [
                        ("🎁", "Loyalty Reward", "Send loyalty points or cashback"),
                        ("🛡️", "Free Trial",     "1-month free Online Security"),
                        ("📧", "Email Campaign", "Enroll in retention email sequence"),
                    ],
                    "Low": [
                        ("📰", "Newsletter",       "Include in monthly newsletter"),
                        ("⭐", "Referral Program", "Invite to loyalty/referral program"),
                    ]
                }
                action_rows = "".join([f"""
                <div style="display:flex;align-items:flex-start;gap:0.6rem;padding:0.45rem 0;border-bottom:1px solid #1f1e1c;">
                    <span style="font-size:1rem;margin-top:1px">{em}</span>
                    <div>
                        <div style="font-size:0.8rem;font-weight:600;color:#cdccca">{title}</div>
                        <div style="font-size:0.75rem;color:#797876;margin-top:1px">{desc}</div>
                    </div>
                </div>""" for em, title, desc in retention[risk]])
                st.markdown(
                    f'<div class="panel"><div class="panel-title">💼 Retention Actions</div>{action_rows}</div>',
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Error: {e}")