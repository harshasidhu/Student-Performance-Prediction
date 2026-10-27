
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "models" / "student_grade_model_4features.joblib"


# -----------------------------------------
# LOAD EXISTING MODEL
# -----------------------------------------
@st.cache_resource
def load_model():
    saved_data = joblib.load(MODEL_PATH)
    return saved_data["model"]


# -----------------------------------------
# CUSTOM STYLING
# -----------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 750;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #dbe4f0;
        background-color: #f8fafc;
        margin-top: 1rem;
    }

    .result-label {
        font-size: 1rem;
        color: #475569;
    }

    .result-score {
        font-size: 3rem;
        font-weight: 750;
        color: #2563eb;
        margin: 0.3rem 0;
    }

    .small-note {
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------
# SIDEBAR
# -----------------------------------------
with st.sidebar:
    st.title("🎓 Grade Predictor")
    st.caption("Student Performance Prediction")

    st.divider()

    st.subheader("About this project")
    st.write(
        "This machine learning application estimates "
        "Portuguese final grades from earlier grades "
        "and student attendance information."
    )

    st.subheader("Model information")
    st.write("**Algorithm:** Extra Trees Regressor")
    st.write("**Target:** G3 — Final grade")
    st.write("**Scale:** 0–20")
    st.write("**Input features:** 4")

    st.divider()

    st.caption(
        "The model was trained on the Portuguese student "
        "dataset. Results may differ for other populations "
        "or educational systems."
    )


# -----------------------------------------
# MAIN HEADER
# -----------------------------------------
st.markdown(
    '<div class="main-title">🎓 Student Performance Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Estimate a student’s Portuguese final grade using four inputs.'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the student's first-period grade, second-period grade, "
    "absences, and previous class failures. The prediction is an "
    "estimate, not a guaranteed final result."
)


# -----------------------------------------
# INPUT FORM
# -----------------------------------------
st.subheader("📝 Student Details")

with st.container(border=True):
    with st.form("student_prediction_form"):
        left, right = st.columns(2, gap="large")

        with left:
            g1 = st.number_input(
                "G1 — First-period grade",
                min_value=0,
                max_value=20,
                value=12,
                step=1,
                help="Enter a grade between 0 and 20."
            )

            g2 = st.number_input(
                "G2 — Second-period grade",
                min_value=0,
                max_value=20,
                value=13,
                step=1,
                help="Enter a grade between 0 and 20."
            )

        with right:
            absences = st.number_input(
                "Absences",
                min_value=0,
                max_value=100,
                value=2,
                step=1,
                help="Enter the number of absences."
            )

            failures = st.number_input(
                "Previous class failures",
                min_value=0,
                max_value=4,
                value=0,
                step=1,
                help="Enter a value from 0 to 4."
            )

        submitted = st.form_submit_button(
            "🔍 Predict Final Grade",
            use_container_width=True,
            type="primary"
        )


# -----------------------------------------
# PREDICTION
# -----------------------------------------
if submitted:
    try:
        model = load_model()

        # Keep feature names and order identical to training.
        student_data = pd.DataFrame(
            [[g1, g2, absences, failures]],
            columns=["G1", "G2", "absences", "failures"]
        )

        prediction = float(model.predict(student_data)[0])

        # Keep the prediction available after subsequent reruns.
        st.session_state["prediction"] = prediction
        st.session_state["student_inputs"] = {
            "G1": g1,
            "G2": g2,
            "Absences": absences,
            "Previous failures": failures
        }

    except FileNotFoundError:
        st.error(
            "The saved model was not found. Check that the file "
            "exists inside the models folder."
        )

    except Exception as error:
        st.error(f"Unable to generate prediction: {error}")


# -----------------------------------------
# DISPLAY RESULTS
# -----------------------------------------
if "prediction" in st.session_state:
    prediction = st.session_state["prediction"]
    student_inputs = st.session_state["student_inputs"]

    st.divider()
    st.subheader("📊 Prediction Results")

    with st.container(border=True):
        st.markdown(
            '<div class="result-label">'
            'Estimated Portuguese Final Grade'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-score">{prediction:.1f} / 20</div>',
            unsafe_allow_html=True
        )

        st.progress(
            min(max(prediction / 20, 0.0), 1.0),
            text="Position on the 0–20 grade scale"
        )

        st.caption(
            "The progress bar represents the numerical score, "
            "not the probability that the prediction is correct."
        )

    st.subheader("Student Input Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("G1", student_inputs["G1"])
    col2.metric("G2", student_inputs["G2"])
    col3.metric("Absences", student_inputs["Absences"])
    col4.metric(
        "Previous failures",
        student_inputs["Previous failures"]
    )

    st.warning(
        "This prediction is an estimate based on historical data. "
        "Actual final grades may differ. Do not use this estimate "
        "as the sole basis for important educational decisions."
    )

else:
    st.divider()
    st.subheader("📊 Prediction Results")
    st.write(
        "Enter the student details above and click "
        "**Predict Final Grade** to view the estimate here."
    )

st.divider()

st.caption(
    "Student Performance Prediction Project | "
    "Portuguese dataset | Extra Trees Regression"
)
