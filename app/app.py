"""
app/app.py
----------
Streamlit application for the Heart Disease Prediction project.

DISCLAIMER: This application is for educational and research purposes only.
It is NOT a medical diagnostic system and must NOT be used for clinical decisions.

Run:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

# Make sure src/ is importable for preprocessing constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_model.pkl"


# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Please run train_models.py first.")
        st.stop()
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("❤️ Heart Disease Prediction")
st.markdown("**Educational / Research Prototype — Not a Medical Diagnostic Tool**")

# ---------------------------------------------------------------------------
# Disclaimer (prominent)
# ---------------------------------------------------------------------------
st.warning(
    "⚠️ **DISCLAIMER**: This application is built for **educational and research purposes only**. "
    "It does **not** constitute a medical diagnosis. Predictions made by this tool should **never** "
    "replace professional medical advice, diagnosis, or treatment. "
    "If you have concerns about your heart health, please consult a qualified healthcare professional."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — About
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Dataset**: UCI Heart Disease (multi-centre)  
    **Records**: 920 patients  
    **Target**: Binary — Heart Disease Present (1) / Absent (0)  
    
    **Models trained**:
    - Logistic Regression
    - K-Nearest Neighbors
    - Decision Tree
    - Random Forest
    - Support Vector Machine
    - Gradient Boosting
    - XGBoost
    
    **Best model** is saved and loaded here.
    """)
    st.markdown("---")
    st.caption("Built with Scikit-learn, XGBoost, Streamlit")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
pipeline = load_model()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("📋 Enter Patient Features")
st.markdown(
    "Fill in the patient's clinical measurements below. "
    "All fields accept the values as they would appear in a clinical record."
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (years)", min_value=1, max_value=120, value=55,
        help="Patient's age in years."
    )
    sex = st.selectbox(
        "Sex", options=["Male", "Female"],
        help="Patient's biological sex."
    )
    cp = st.selectbox(
        "Chest Pain Type",
        options=["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
        help="Type of chest pain reported by the patient."
    )
    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50.0, max_value=250.0, value=130.0, step=1.0,
        help="Resting blood pressure on admission (mm Hg)."
    )
    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=50.0, max_value=600.0, value=240.0, step=1.0,
        help="Serum cholesterol level in mg/dl."
    )
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=["False", "True"],
        help="Whether fasting blood sugar > 120 mg/dl (True = yes)."
    )
    restecg = st.selectbox(
        "Resting ECG Result",
        options=["normal", "lv hypertrophy", "st-t abnormality"],
        help="Resting electrocardiographic results."
    )

with col2:
    thalch = st.number_input(
        "Max Heart Rate Achieved",
        min_value=50.0, max_value=250.0, value=150.0, step=1.0,
        help="Maximum heart rate achieved during exercise test."
    )
    exang = st.selectbox(
        "Exercise-Induced Angina",
        options=["False", "True"],
        help="Whether exercise induced angina (True = yes)."
    )
    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0, max_value=10.0, value=1.0, step=0.1,
        help="ST depression induced by exercise relative to rest."
    )
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=["upsloping", "flat", "downsloping"],
        help="The slope of the peak exercise ST segment."
    )
    ca = st.number_input(
        "Number of Major Vessels (0–3)",
        min_value=0.0, max_value=3.0, value=0.0, step=1.0,
        help="Number of major vessels coloured by fluoroscopy (0–3)."
    )
    thal = st.selectbox(
        "Thalassemia Type",
        options=["normal", "fixed defect", "reversable defect"],
        help="Results of thallium stress test."
    )

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
st.markdown("---")
if st.button("🔍 Predict Heart Disease Risk", use_container_width=True, type="primary"):

    # Build a single-row DataFrame with the exact columns the pipeline expects
    input_data = pd.DataFrame([{
        "age": float(age),
        "sex": sex,
        "cp": cp,
        "trestbps": float(trestbps),
        "chol": float(chol),
        "fbs": fbs,
        "restecg": restecg,
        "thalch": float(thalch),
        "exang": exang,
        "oldpeak": float(oldpeak),
        "slope": slope,
        "ca": float(ca),
        "thal": thal,
    }])

    try:
        prediction = pipeline.predict(input_data)[0]
        proba = pipeline.predict_proba(input_data)[0]
        prob_disease = proba[1]
        prob_healthy = proba[0]

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        if prediction == 1:
            st.error(
                f"**Prediction: Heart Disease Detected (Class 1)**  \n"
                f"Estimated probability of heart disease: **{prob_disease:.1%}**"
            )
        else:
            st.success(
                f"**Prediction: No Heart Disease Detected (Class 0)**  \n"
                f"Estimated probability of heart disease: **{prob_disease:.1%}**"
            )

        # Probability gauge
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("P(No Disease)", f"{prob_healthy:.1%}")
        with col_b:
            st.metric("P(Disease)", f"{prob_disease:.1%}")

        st.progress(float(prob_disease))

        st.markdown("---")
        st.info(
            "🔬 **Research Note**: This prediction is generated by a machine learning "
            "model trained on the UCI Heart Disease dataset. The model has not been "
            "clinically validated. Probability values reflect model confidence, "
            "not actual medical probability. **Always consult a physician.**"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.exception(e)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Heart Disease Prediction · Educational Portfolio Project · "
    "Data: UCI Heart Disease Repository · Not for clinical use."
)
