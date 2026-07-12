"""
app.py
------
Streamlit demo for Endometriosis Detection using Data Mining Techniques.

Run locally with:
    streamlit run app.py

Loads the model trained by train_model.py (best_model.pkl + scaler.pkl)
and lets a user enter patient symptom/clinical data to get a real-time
prediction with a confidence score.
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Endometriosis Detection",
    page_icon="🩺",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Load trained artifacts (cached so they load only once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_list.json") as f:
        features = json.load(f)
    try:
        with open("model_comparison.json") as f:
            comparison = json.load(f)
    except FileNotFoundError:
        comparison = None
    return model, scaler, features, comparison


model, scaler, FEATURES, comparison = load_artifacts()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🩺 Endometriosis Detection")
st.caption(
    "A data mining–based screening aid that predicts endometriosis risk "
    "from patient symptom and clinical data using classification algorithms."
)

if comparison:
    st.info(
        f"**Model in use:** {comparison['best_model']}  |  "
        f"Test Recall: {comparison['results'][comparison['best_model']]['test_recall']}  |  "
        f"Test Accuracy: {comparison['results'][comparison['best_model']]['test_accuracy']}"
    )

st.divider()

# ---------------------------------------------------------------------------
# Sidebar: model comparison table (nice for a live demo / interview)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Model Comparison")
    if comparison:
        df_results = pd.DataFrame(comparison["results"]).T
        df_results = df_results[
            ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
        ]
        df_results.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        st.dataframe(df_results.style.highlight_max(axis=0, color="#d4f7dc"))
        st.caption(
            f"Best model selected by **Recall** (medical context: "
            f"false negatives are costlier than false positives) → "
            f"**{comparison['best_model']}**"
        )
    else:
        st.write("Run `train_model.py` first to generate model_comparison.json")

    st.divider()
    st.header("ℹ️ About")
    st.write(
        "Built with Python, scikit-learn, and Streamlit. "
        "Pipeline: data cleaning → feature scaling → class balancing → "
        "5-algorithm comparison → GridSearchCV + cross-validation tuning → "
        "best-model deployment."
    )

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 50, 28)
    bmi = st.slider("BMI", 15.0, 40.0, 24.0, step=0.1)
    pelvic_pain_severity = st.slider("Pelvic Pain Severity (0 = none, 10 = severe)", 0, 10, 4)
    ca125_level = st.slider("CA-125 Level (U/mL)", 5.0, 80.0, 28.0, step=0.5)
    cycle_irregularity = st.radio("Irregular Menstrual Cycle?", ["No", "Yes"], horizontal=True)

with col2:
    infertility_history = st.radio("History of Infertility?", ["No", "Yes"], horizontal=True)
    family_history = st.radio("Family History of Endometriosis?", ["No", "Yes"], horizontal=True)
    painful_intercourse = st.radio("Painful Intercourse?", ["No", "Yes"], horizontal=True)
    heavy_bleeding = st.radio("Heavy Menstrual Bleeding?", ["No", "Yes"], horizontal=True)
    chronic_fatigue = st.radio("Chronic Fatigue?", ["No", "Yes"], horizontal=True)

yn = {"No": 0, "Yes": 1}

input_dict = {
    "age": age,
    "bmi": bmi,
    "pelvic_pain_severity": pelvic_pain_severity,
    "cycle_irregularity": yn[cycle_irregularity],
    "infertility_history": yn[infertility_history],
    "family_history": yn[family_history],
    "painful_intercourse": yn[painful_intercourse],
    "heavy_bleeding": yn[heavy_bleeding],
    "chronic_fatigue": yn[chronic_fatigue],
    "ca125_level": ca125_level,
}

input_df = pd.DataFrame([input_dict])[FEATURES]  # enforce correct column order

st.divider()

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ **Likely Endometriosis** — model confidence: {probability:.1%}")
        st.write(
            "This result suggests the patient's symptom profile is consistent with "
            "endometriosis. **This is a screening aid, not a diagnosis** — "
            "recommend referral for clinical/imaging confirmation "
            "(e.g., transvaginal ultrasound, MRI, or laparoscopy)."
        )
    else:
        st.success(f"✅ **Low Likelihood of Endometriosis** — model confidence: {(1 - probability):.1%}")
        st.write(
            "Symptom profile does not strongly match typical endometriosis patterns. "
            "Continue routine monitoring; re-assess if symptoms change."
        )

    st.progress(float(probability))
    st.caption(f"Predicted probability of endometriosis: {probability:.1%}")

    # Feature importance / top contributing symptoms (if model supports it)
    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        imp_df = pd.DataFrame({"Feature": FEATURES, "Influence": importances})
        imp_df = imp_df.sort_values("Influence", ascending=False).head(5)
        st.write("**Top contributing factors (this model):**")
        st.bar_chart(imp_df.set_index("Feature"))
    elif hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame(
            {"Feature": FEATURES, "Influence": model.feature_importances_}
        ).sort_values("Influence", ascending=False).head(5)
        st.write("**Top contributing factors (this model):**")
        st.bar_chart(imp_df.set_index("Feature"))

st.divider()
st.caption(
    "⚕️ Disclaimer: This tool is a data-driven screening aid built for an academic/portfolio "
    "project. It is not a certified medical device and should not replace professional "
    "diagnosis by a qualified healthcare provider."
)
