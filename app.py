"""
app.py
------
Streamlit demo for Endometriosis Detection using Data Mining Techniques.

Run locally with:
    streamlit run app.py

Loads the model trained by train_model.py (best_model.pkl + scaler.pkl)
and lets a user enter patient symptom/clinical data to get a real-time
prediction with a confidence score.

Color palette: plum / lavender / teal (calming, clinical, not clinical-white)
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
# Custom CSS — plum / lavender / teal palette
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Overall app background — soft lavender wash */
    .stApp {
        background: linear-gradient(180deg, #F6F2FB 0%, #EFE7FA 45%, #E7F4F1 100%);
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #6E3F99 0%, #8C5BB5 50%, #3F9E8F 100%);
        padding: 2rem 2rem 1.6rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(110, 63, 153, 0.25);
    }
    .hero-banner h1 {
        color: #FFFFFF !important;
        margin-bottom: 0.3rem;
        font-size: 2rem;
    }
    .hero-banner p {
        color: #EDE3FA !important;
        font-size: 1rem;
        margin: 0;
    }

    /* Section headers */
    h3 {
        color: #5A2E82 !important;
        border-left: 5px solid #3F9E8F;
        padding-left: 0.6rem;
        margin-top: 1.5rem !important;
    }

    /* Input containers -> card look */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 0.5rem;
        box-shadow: 0 2px 10px rgba(90, 46, 130, 0.08);
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #6E3F99, #3F9E8F);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(110, 63, 153, 0.35);
        color: white;
    }

    /* Result banners */
    .result-positive {
        background: linear-gradient(135deg, #FDEBEC 0%, #FBD9DC 100%);
        border-left: 6px solid #C3455B;
        padding: 1.1rem 1.3rem;
        border-radius: 12px;
        margin-top: 0.6rem;
    }
    .result-positive h4 { color: #A22E45 !important; margin: 0 0 0.4rem 0; }
    .result-positive p { color: #7A2C3B; margin: 0; }

    .result-negative {
        background: linear-gradient(135deg, #E7F6F1 0%, #D6F0E6 100%);
        border-left: 6px solid #2E9E7A;
        padding: 1.1rem 1.3rem;
        border-radius: 12px;
        margin-top: 0.6rem;
    }
    .result-negative h4 { color: #1F7A5C !important; margin: 0 0 0.4rem 0; }
    .result-negative p { color: #1F5C48; margin: 0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3F2A57 0%, #2C3E4A 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #F1EAFB !important;
    }
    section[data-testid="stSidebar"] .stDataFrame {
        background-color: #FFFFFF;
        border-radius: 8px;
    }

    /* Disclaimer footer */
    .disclaimer-box {
        background-color: #FFF7E8;
        border-left: 5px solid #D9A441;
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        color: #7A5A1E;
        font-size: 0.88rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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
# Header (custom gradient banner instead of plain st.title)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>🩺 Endometriosis Detection</h1>
    <p>A data mining–based screening aid that predicts endometriosis risk from
    patient symptom and clinical data using classification algorithms.</p>
</div>
""", unsafe_allow_html=True)

if comparison:
    best = comparison['best_model']
    r = comparison['results'][best]
    st.info(
        f"**Model in use:** {best}  |  "
        f"Test Recall: {r['test_recall']}  |  "
        f"Test Accuracy: {r['test_accuracy']}"
    )

# ---------------------------------------------------------------------------
# Sidebar: model comparison table
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Model Comparison")
    if comparison:
        df_results = pd.DataFrame(comparison["results"]).T
        df_results = df_results[
            ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
        ]
        df_results.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        st.dataframe(df_results.style.highlight_max(axis=0, color="#B8E8D4"))
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
st.markdown("### Enter Patient Details")

with st.container(border=True):
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

st.write("")

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.markdown(f"""
        <div class="result-positive">
            <h4>⚠️ Likely Endometriosis — confidence: {probability:.1%}</h4>
            <p>This result suggests the patient's symptom profile is consistent with
            endometriosis. This is a screening aid, not a diagnosis — recommend
            referral for clinical/imaging confirmation (e.g., transvaginal
            ultrasound, MRI, or laparoscopy).</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <h4>✅ Low Likelihood of Endometriosis — confidence: {(1 - probability):.1%}</h4>
            <p>Symptom profile does not strongly match typical endometriosis
            patterns. Continue routine monitoring; re-assess if symptoms change.</p>
        </div>
        """, unsafe_allow_html=True)

    st.progress(float(probability))
    st.caption(f"Predicted probability of endometriosis: {probability:.1%}")

    # Feature importance / top contributing symptoms (if model supports it)
    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        imp_df = pd.DataFrame({"Feature": FEATURES, "Influence": importances})
        imp_df = imp_df.sort_values("Influence", ascending=False).head(5)
        st.write("**Top contributing factors (this model):**")
        st.bar_chart(imp_df.set_index("Feature"), color="#7A4DA0")
    elif hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame(
            {"Feature": FEATURES, "Influence": model.feature_importances_}
        ).sort_values("Influence", ascending=False).head(5)
        st.write("**Top contributing factors (this model):**")
        st.bar_chart(imp_df.set_index("Feature"), color="#7A4DA0")

st.markdown("""
<div class="disclaimer-box">
    ⚕️ <strong>Disclaimer:</strong> This tool is a data-driven screening aid built for an
    academic/portfolio project. It is not a certified medical device and should not
    replace professional diagnosis by a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)
