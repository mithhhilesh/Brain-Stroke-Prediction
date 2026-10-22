from datetime import datetime
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Brain Stroke Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
SEARCH_DIRS = [BASE_DIR / "assets", BASE_DIR, BASE_DIR / "assets" / "assets"]

TEAL, CYAN, INDIGO, AMBER, ROSE, SLATE = (
    "#0f766e", "#0e7490", "#4338ca", "#f59e0b", "#be185d", "#64748b",
)
PALETTE = [TEAL, CYAN, INDIGO, AMBER, ROSE, SLATE]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def find_asset(filename):
    """Find a results file in assets/, next to app.py, or in assets/assets/."""
    for folder in SEARCH_DIRS:
        if (folder / filename).exists():
            return folder / filename
    stem, ext = Path(filename).stem, Path(filename).suffix
    for folder in SEARCH_DIRS:  # tolerate renames like "name (1).csv"
        if folder.exists():
            matches = sorted(folder.glob(f"{stem}*{ext}"))
            if matches:
                return matches[0]
    return None


def read_csv_if_exists(filename):
    path = find_asset(filename)
    return pd.read_csv(path) if path else None


def wide(fn, *args, **kwargs):
    if fn == st.image:
        return fn(*args, use_column_width=True, **kwargs)
    return fn(*args, use_container_width=True, **kwargs)

def show_chart(chart):
    chart = chart.configure_view(strokeWidth=0).configure_axis(
        labelFontSize=12, titleFontSize=12, gridColor="#e3edf1",
        domainColor="#c3d3da", labelColor="#33505c", titleColor="#33505c",
    ).configure_legend(labelFontSize=12, titleFontSize=12, orient="bottom")
    return st.altair_chart(chart, use_container_width=True, theme=None)


def show_image_if_exists(filename, caption):
    path = find_asset(filename)
    if path:
        wide(st.image, str(path), caption=caption)
        return True
    return False


# ------------------------------------------------------------------
# Styling (UI only - no effect on the model or predictions)
# ------------------------------------------------------------------
st.markdown(
    """
<style>
:root { color-scheme: light; --ink:#162b3a; --muted:#526776; --line:#dbe5e9; --teal:#087f80; }
html, body, .stApp { font-family: 'Segoe UI', Arial, sans-serif; color:var(--ink); }
.stApp { background:#f3f6f8; }
[data-testid="stAppViewContainer"] > .main { background:#f3f6f8; }
[data-testid="stHeader"] { background:#f3f6f8; }
#MainMenu, footer { visibility:hidden; }
.block-container { max-width:1160px; padding:3.3rem 2.4rem 3rem; }
h1, h2, h3, h4, p, label, .stMarkdown { color:var(--ink); }
h2 { font-size:1.55rem !important; font-weight:700 !important; letter-spacing:-.03em; }
h3 { font-size:1.18rem !important; font-weight:650 !important; letter-spacing:-.02em; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:var(--muted) !important; }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top:2rem; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color:var(--ink) !important; }
[data-testid="stSidebar"] code { color:#086a6b; background:#e9f5f3; }
.sidebar-brand { padding:0 0 1.25rem; margin-bottom:1rem; border-bottom:1px solid var(--line); }
.sidebar-brand strong { display:block; font-size:1.4rem; letter-spacing:-.04em; }
.sidebar-brand span { color:var(--muted); font-size:.82rem; }
.brand-mark { display:inline-flex; align-items:center; justify-content:center; width:38px; height:38px;
  color:#fff; background:var(--teal); border-radius:11px; font-size:1.7rem; margin-bottom:.7rem; }
.hero { display:flex; align-items:center; justify-content:space-between; gap:2rem;
  padding:2rem 2.1rem; background:#123846; border:1px solid #214957; border-radius:18px; margin-bottom:1rem; }
.hero .eyebrow { color:#9edfd7; text-transform:uppercase; font-size:.72rem; font-weight:700;
  letter-spacing:.15em; margin:0 0 .6rem; }
.hero h1 { color:#fff; font-size:clamp(1.75rem,3vw,2.45rem); font-weight:700; line-height:1.14;
  letter-spacing:-.045em; margin:0; padding:0; }
.hero .intro { color:#d1e3e8; font-size:.95rem; line-height:1.6; max-width:52ch; margin:.8rem 0 0; }
.hero-status { min-width:195px; padding:1rem 1.2rem; border:1px solid #41616c; border-radius:12px; }
.hero-status span { display:block; color:#a9c5cd; font-size:.72rem; margin-bottom:.4rem; }
.hero-status strong { display:block; color:#fff; font-size:.95rem; }
.hero-status small { display:block; color:#c5dde3; font-size:.78rem; margin-top:.5rem; }
.notice { font-size:.82rem; color:#506773; margin:.3rem 0 1.6rem; line-height:1.6; }
.stTabs [data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid var(--line); padding-bottom:.45rem;
  overflow-x:auto; scrollbar-width:thin; }
.stTabs [data-baseweb="tab"] { color:#506573; background:transparent; border-radius:8px;
  padding:.6rem .9rem; height:auto; white-space:nowrap; }
.stTabs [data-baseweb="tab"] p { color:inherit; font-size:.88rem; font-weight:600; }
.stTabs [aria-selected="true"] { background:#e0f1ed !important; color:#076c68 !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
[data-testid="stTabContent"] { padding-top:1.25rem; }
[data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:14px;
  padding:1.6rem 1.6rem 1.2rem; box-shadow:0 3px 12px #172f3c05; }
.group-title { font-size:.78rem; font-weight:750; text-transform:uppercase; letter-spacing:.08em;
  color:#14706c; padding-bottom:.65rem; border-bottom:1px solid #e5eeed; margin:0 0 .5rem; }
[data-testid="stWidgetLabel"] p { color:#2c4553 !important; font-size:.86rem; font-weight:550; }
[data-baseweb="select"] > div { background:#f8fafb !important; border-color:#ccdbe1 !important; border-radius:8px; }
[data-baseweb="select"] span, [data-baseweb="select"] input { color:#173747 !important; }
[data-baseweb="input"], [data-baseweb="base-input"] { background:#f8fafb !important; border-color:#ccdbe1 !important; }
[data-baseweb="input"] input { color:#173747 !important; -webkit-text-fill-color:#173747; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul { background:#fff !important; color:#173747 !important; }
[data-baseweb="popover"] li { color:#173747 !important; }
[data-testid="stNumberInput"] button { color:#2c5665 !important; background:#eef4f6 !important; }
[data-testid="stSlider"] [data-testid="stTickBar"] { color:#3c5967 !important; }
div.stButton > button, [data-testid="stDownloadButton"] button {
  background:#fff; color:#205362; border:1px solid #c9dadf; border-radius:8px;
  padding:.5rem .9rem; min-height:2.6rem; width:100%; box-shadow:none; }
div.stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
  background:#eaf5f2; border-color:#509f97; color:#075e59; }
div.stButton > button p, [data-testid="stDownloadButton"] button p { color:inherit; font-size:.85rem; }
[data-testid="stFormSubmitButton"] button {
  width:100%; min-height:3rem; border:1px solid #087b76; border-radius:9px;
  background:#087b76; color:#fff; margin-top:.6rem; }
[data-testid="stFormSubmitButton"] button p { color:#fff !important; font-weight:650; }
[data-testid="stFormSubmitButton"] button:hover { background:#06615e; border-color:#06615e; }
button:focus-visible, input:focus-visible { outline:3px solid #48b8ab !important; outline-offset:2px; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:1rem 1.15rem; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color:var(--ink); }
[data-testid="stMetricValue"] { font-size:1.8rem; font-weight:650; }
.result-card { padding:1.25rem 1.45rem; margin:.5rem 0 1rem; border:1px solid; border-radius:12px; }
.result-card .label { color:#183c46; font-size:1.3rem; font-weight:650; margin:0; }
.result-card .sub { color:#465e65; font-size:.88rem; margin:.5rem 0 0; line-height:1.6; }
.result-neg { background:#eaf5ef; border-color:#badbc8; }
.result-pos { background:#fff4df; border-color:#e7cd9a; }
.gauge { position:relative; height:12px; border-radius:20px; background:#e1e8ec; margin:1rem 0 .5rem; }
.gauge-fill { height:100%; border-radius:20px; background:#179c91; }
.gauge-marker { position:absolute; top:-5px; width:3px; height:22px; background:#173a47; border-radius:2px; }
.gauge-legend { display:flex; justify-content:space-between; color:#516976; font-size:.76rem; margin-bottom:1.4rem; }
.steps { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.8rem; margin:.8rem 0 1.6rem; }
.step { background:#fff; border:1px solid var(--line); border-radius:12px; padding:1.15rem; }
.step .num { display:inline-flex; align-items:center; justify-content:center; width:25px; height:25px;
  background:#e3f2ee; color:#0a766d; border-radius:7px; margin-right:.5rem; font-size:.8rem; font-weight:700; }
.step b { font-size:.92rem; color:#244653; }
.step p { color:#5a707b; font-size:.85rem; line-height:1.55; margin:.6rem 0 0; }
[data-testid="stExpander"] { background:#fff; border:1px solid var(--line); border-radius:10px; }
[data-testid="stExpander"] summary { color:var(--ink); }
[data-testid="stAlert"] { border-radius:10px; }
[data-testid="stVegaLiteChart"] { background:#fff; border-radius:12px; padding:.4rem; border:1px solid #e0e8eb; }
@media(max-width:760px) {
  .block-container { padding:3rem 1rem 2rem; }
  .hero { display:block; padding:1.4rem; border-radius:12px; }
  .hero-status { margin-top:1.2rem; min-width:0; }
  .hero h1 { font-size:1.85rem; }
  [data-testid="stForm"] { padding:1rem; }
  .steps { grid-template-columns:1fr; }
  .stTabs [data-baseweb="tab"] { padding:.6rem .7rem; }
}
@media(prefers-reduced-motion:reduce) { *,*::before,*::after { animation:none !important; transition:none !important; } }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Load the saved model and its input configuration
# ------------------------------------------------------------------
MODEL_FILES = {
    "Logistic Regression": "stroke_lr.joblib",
    "Decision Tree": "stroke_dt.joblib",
    "Random Forest": "stroke_rf.joblib",
}
st.sidebar.markdown(
    '<div class="sidebar-brand"><div class="brand-mark">+</div>'
    '<strong>Brain Stroke</strong><span>Machine learning project</span></div>',
    unsafe_allow_html=True,
)
selected_model = st.sidebar.selectbox("Prediction model", list(MODEL_FILES))
model_path = BASE_DIR / "model_exports" / MODEL_FILES[selected_model]

@st.cache_resource
def load_artifact(path, modified_ns):
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("artifact_version") != 2:
        raise ValueError("Re-export this model using the corrected notebook.")
    return artifact

if not model_path.is_file():
    st.error("The saved model is missing. Run the notebook from start to finish to create it.")
    st.stop()
artifact = load_artifact(str(model_path), model_path.stat().st_mtime_ns)
model = artifact["model"]
feature_columns = artifact["encoded_feature_columns"]
input_columns = artifact["input_columns"]
threshold = artifact["threshold"]
positive_index = artifact["classes"].index(artifact["positive_class"])
if st.session_state.get("result_model") != selected_model:
    st.session_state.pop("last_result", None)
    st.session_state["result_model"] = selected_model

def final_estimator(m):
    while hasattr(m, "steps"):
        m = m.steps[-1][1]
    return m

model_name = type(final_estimator(model)).__name__


def get_importances(m, columns):
    est = final_estimator(m)
    if hasattr(est, "feature_importances_"):
        values, label = est.feature_importances_, "Feature importance"
    elif hasattr(est, "coef_"):
        values, label = np.abs(np.ravel(est.coef_)), "Absolute coefficient"
    else:
        return None, None
    if len(values) != len(columns):
        return None, None
    series = pd.Series(values, index=[c.replace("_", " ") for c in columns])
    return series.sort_values(ascending=False), label


# ------------------------------------------------------------------
# Same preprocessing as the original app, wrapped in two small functions
# ------------------------------------------------------------------
def raw_row(v):
    return {
        "gender": v["gender"],
        "age": v["age"],
        "hypertension": 1 if v["hypertension"] == "Yes" else 0,
        "heart_disease": 1 if v["heart_disease"] == "Yes" else 0,
        "ever_married": v["ever_married"],
        "work_type": v["work_type"],
        "Residence_type": v["residence_type"],
        "avg_glucose_level": v["avg_glucose_level"],
        "bmi": v["bmi"],
        "smoking_status": v["smoking_status"],
    }


def encode(rows):
    # The saved pipeline owns all encoding and feature ordering.
    return pd.DataFrame(rows).loc[:, input_columns]


# ------------------------------------------------------------------
# Form defaults and example profiles (illustrative, not real patients)
# ------------------------------------------------------------------
DEFAULTS = {
    "gender": "Male", "age": 45, "hypertension": "No", "heart_disease": "No",
    "ever_married": "No", "work_type": "Private", "residence_type": "Urban",
    "smoking_status": "never smoked", "avg_glucose_level": 100.0, "bmi": 25.0,
}
PRESETS = {
    "Younger profile": {
        "gender": "Female", "age": 25, "hypertension": "No", "heart_disease": "No",
        "ever_married": "No", "work_type": "Private", "residence_type": "Urban",
        "smoking_status": "never smoked", "avg_glucose_level": 85.0, "bmi": 22.0,
    },
    "Middle-aged profile": {
        "gender": "Female", "age": 55, "hypertension": "Yes", "heart_disease": "No",
        "ever_married": "Yes", "work_type": "Govt_job", "residence_type": "Urban",
        "smoking_status": "smokes", "avg_glucose_level": 140.0, "bmi": 29.5,
    },
    "Older profile": {
        "gender": "Male", "age": 72, "hypertension": "Yes", "heart_disease": "Yes",
        "ever_married": "Yes", "work_type": "Self-employed", "residence_type": "Rural",
        "smoking_status": "formerly smoked", "avg_glucose_level": 210.0, "bmi": 31.0,
    },
    "Reset": DEFAULTS,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("history", [])


def apply_preset(name):
    st.session_state.update(PRESETS[name])
    st.session_state.pop("last_result", None)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.subheader("Project details")
    st.write(
        "Explore three classification models trained on the same brain-stroke dataset."
    )
    st.markdown(
        f"""
        **Dataset**
        - 4,981 records, 11 columns
        - 80:20 train-test split

        **Saved model in this app**
        - `{model_name}`
        - Threshold: {threshold}
        """
    )
    st.divider()
    st.caption("Educational use only. Not a clinically validated diagnostic system.")

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <div>
        <p class="eyebrow">Explore · Compare · Understand</p>
        <h1>Brain Stroke Prediction</h1>
        <p class="intro">Explore how patient details influence a trained model's
        prediction. Compare three approaches and understand their results.</p>
      </div>
      <div class="hero-status">
        <span>ACTIVE MODEL</span>
        <strong>{selected_model}</strong>
        <small>Decision threshold · {threshold:.0%}</small>
      </div>
    </div>
    <p class="notice">Educational demonstration. Predictions describe model behavior
    and are not a medical diagnosis or a validated estimate of personal risk.</p>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_overview, tab_compare, tab_threshold, tab_explain, tab_eda = st.tabs(
    ["Prediction", "Project overview", "Compare models", "Threshold study", "Feature insights", "Data & evaluation"]
)

# ==================================================================
# PREDICT (form, result, what-if analysis, download, history)
# ==================================================================
with tab_predict:
    st.subheader("Patient details")
    st.caption("Start with an example or enter your own values. Example profiles are fictional.")
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, PRESETS):
        col.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name,))

    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<p class="group-title">Personal profile</p>', unsafe_allow_html=True)
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
            age = st.slider("Age (years)", 0, 100, key="age")
            ever_married = st.selectbox("Ever married", ["No", "Yes"], key="ever_married")
            work_type = st.selectbox(
                "Work type",
                ["Private", "Self-employed", "Govt_job", "children"],
                key="work_type",
            )
            residence_type = st.selectbox("Residence type", ["Urban", "Rural"], key="residence_type")
        with col2:
            st.markdown('<p class="group-title">Health and lifestyle</p>', unsafe_allow_html=True)
            hypertension = st.selectbox("Hypertension", ["No", "Yes"], key="hypertension")
            heart_disease = st.selectbox("Heart disease", ["No", "Yes"], key="heart_disease")
            smoking_status = st.selectbox(
                "Smoking status",
                ["never smoked", "formerly smoked", "smokes", "Unknown"],
                key="smoking_status",
            )
            avg_glucose_level = st.number_input(
                "Average glucose level (mg/dL)", min_value=0.1, max_value=400.0,
                step=1.0, key="avg_glucose_level",
            )
            bmi = st.number_input(
                "BMI (kg/m²)", min_value=0.1, max_value=100.0, step=0.1, key="bmi",
            )
        submitted = st.form_submit_button("Predict stroke class")

    if submitted:
        vals = {
            "gender": gender, "age": age, "hypertension": hypertension,
            "heart_disease": heart_disease, "ever_married": ever_married,
            "work_type": work_type, "residence_type": residence_type,
            "smoking_status": smoking_status,
            "avg_glucose_level": avg_glucose_level, "bmi": bmi,
        }
        probability = float(model.predict_proba(encode([raw_row(vals)]))[0][positive_index])
        prediction = int(probability >= threshold)
        st.session_state["last_result"] = {
            "vals": vals, "probability": probability, "prediction": prediction,
        }
        st.session_state["history"].append({
            "Model": selected_model,
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Age": age, "Gender": gender, "Hypertension": hypertension,
            "Heart disease": heart_disease, "Glucose": avg_glucose_level,
            "BMI": bmi, "Probability": f"{probability:.1%}",
            "Class": "Stroke (1)" if prediction == 1 else "No Stroke (0)",
        })
        st.session_state["history"] = st.session_state["history"][-20:]

    last = st.session_state.get("last_result")
    if last:
        vals = last["vals"]
        probability = last["probability"]
        prediction = last["prediction"]

        st.divider()
        st.subheader("Prediction result")
        st.caption("Based on your last submitted values and the selected model.")
        if prediction == 1:
            st.markdown(
                f"""<div class="result-card result-pos">
                <p class="label">Predicted class: Stroke (1)</p>
                <p class="sub">The model probability of {probability:.1%} is at or above the threshold of {threshold}.</p>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="result-card result-neg">
                <p class="label">Predicted class: No Stroke (0)</p>
                <p class="sub">The model probability of {probability:.1%} is below the threshold of {threshold}.</p>
                </div>""",
                unsafe_allow_html=True,
            )

        m1, m2 = st.columns(2)
        m1.metric("Model probability · stroke class", f"{probability:.1%}")
        m2.metric("Classification threshold", f"{threshold}")

        fill = min(max(probability, 0.0), 1.0) * 100
        marker = min(max(float(threshold), 0.0), 1.0) * 100
        st.markdown(
            f"""
            <div class="gauge">
                <div class="gauge-fill" style="width:{fill}%"></div>
                <div class="gauge-marker" style="left:{marker}%"></div>
            </div>
            <div class="gauge-legend"><span>0%</span>
            <span>Dark marker = threshold ({threshold})</span><span>100%</span></div>
            """,
            unsafe_allow_html=True,
        )

        # ---------- What-if analysis ----------
        st.subheader("What-if analysis")
        st.caption(
            "Change one input while keeping the others fixed and watch how the "
            "model's probability responds. Computed live from the saved model."
        )
        sweeps = {
            "Age (years)": ("age", np.arange(0, 101, 1)),
            "Average glucose level (mg/dL)": ("avg_glucose_level", np.arange(50.0, 300.1, 5.0)),
            "BMI (kg/m²)": ("bmi", np.arange(10.0, 60.1, 0.5)),
        }
        chosen_label = st.selectbox("Feature to vary", list(sweeps), key="whatif_feature")
        feat_key, grid = sweeps[chosen_label]
        rows = []
        for g in grid:
            row = dict(vals)
            row[feat_key] = float(g) if feat_key != "age" else int(g)
            rows.append(raw_row(row))
        probs = model.predict_proba(encode(rows))[:, positive_index]
        sweep_df = pd.DataFrame({"x": grid.astype(float), "Probability": probs})
        current_df = pd.DataFrame(
            {"x": [float(vals[feat_key])], "Probability": [probability]}
        )
        line = alt.Chart(sweep_df).mark_line(color=TEAL, strokeWidth=3).encode(
            x=alt.X("x:Q", title=chosen_label),
            y=alt.Y("Probability:Q", title="Model probability for class 1",
                    scale=alt.Scale(domain=[0, min(1.0, max(0.05, float(threshold) * 1.15, float(probs.max()) * 1.15))]),
                    axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("x:Q", title=chosen_label),
                     alt.Tooltip("Probability:Q", format=".1%")],
        )
        rule = alt.Chart(pd.DataFrame({"t": [float(threshold)]})).mark_rule(
            color=AMBER, strokeDash=[6, 4], strokeWidth=2
        ).encode(y="t:Q")
        point = alt.Chart(current_df).mark_point(
            color=INDIGO, size=170, filled=True
        ).encode(x="x:Q", y="Probability:Q")
        show_chart((line + rule + point).properties(height=280))
        st.caption(
            "Teal line = model probability. Dashed amber line = classification "
            "threshold. Indigo dot = the values you submitted. This shows how the "
            "model behaves, not what would happen to a real person."
        )

        # ---------- Download + inputs ----------
        report = pd.DataFrame([{
            **vals,
            "model": selected_model,
            "probability_class_1": round(probability, 4),
            "threshold": threshold,
            "predicted_class": prediction,
        }])
        st.download_button(
            "Download this result (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="stroke_prediction_result.csv",
            mime="text/csv",
        )
        with st.expander("View submitted inputs"):
            wide(
                st.dataframe,
                pd.DataFrame({
                    "Feature": ["Gender", "Age", "Hypertension", "Heart disease",
                                "Ever married", "Work type", "Residence type",
                                "Average glucose level", "BMI", "Smoking status"],
                    "Value": [vals["gender"], str(vals["age"]), vals["hypertension"],
                              vals["heart_disease"], vals["ever_married"],
                              vals["work_type"], vals["residence_type"],
                              str(vals["avg_glucose_level"]), str(vals["bmi"]),
                              vals["smoking_status"]],
                }),
                hide_index=True,
            )
        st.caption(
            f"Experimental classification threshold: {threshold}. "
            "The displayed probability is not a validated estimate "
            "of an individual's clinical risk."
        )

    if not last:
        st.caption("Your result and an interactive what-if chart will appear here after you submit the form.")

    if st.session_state["history"]:
        with st.expander(f"Prediction history this session ({len(st.session_state['history'])})"):
            wide(st.dataframe, pd.DataFrame(st.session_state["history"][::-1]), hide_index=True)

# ==================================================================
# OVERVIEW
# ==================================================================
with tab_overview:
    st.subheader("Project workflow")
    st.markdown(
        """
        <div class="steps">
          <div class="step"><span class="num">1</span><b>Data</b>
            <p>Brain-stroke dataset with 4,981 records and 11 columns.</p></div>
          <div class="step"><span class="num">2</span><b>Exploration and EDA</b>
            <p>Data exploration and exploratory data analysis.</p></div>
          <div class="step"><span class="num">3</span><b>Preprocessing</b>
            <p>Data preprocessing and categorical encoding.</p></div>
          <div class="step"><span class="num">4</span><b>Train-test split</b>
            <p>80:20 split into training and test sets.</p></div>
          <div class="step"><span class="num">5</span><b>Model training</b>
            <p>Logistic Regression, Decision Tree and Random Forest.</p></div>
          <div class="step"><span class="num">6</span><b>Validation and tuning</b>
            <p>Cross-validation and hyperparameter tuning.</p></div>
          <div class="step"><span class="num">7</span><b>Evaluation</b>
            <p>Confusion matrix, precision, recall, F1-score, accuracy and ROC-AUC.</p></div>
          <div class="step"><span class="num">8</span><b>Threshold tuning</b>
            <p>Random Forest threshold study: recall vs false positives.</p></div>
          <div class="step"><span class="num">9</span><b>Deployment</b>
            <p>This Streamlit app, loading the saved model file.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Limitations")
    st.markdown(
        """
        - The model was trained on a single public dataset of 4,981 records.
        - The output is a class assigned by a threshold, not a diagnosis.
        - The probability shown has not been clinically validated.
        - This project is for learning and demonstration only.
        """
    )

# ==================================================================
# COMPARE (needs assets/metrics_comparison.csv)
# ==================================================================
with tab_compare:
    st.subheader("Model comparison")
    metrics_df = read_csv_if_exists("metrics_comparison.csv")
    if metrics_df is None:
        st.info("metrics_comparison.csv not found. Put it in an `assets` folder next to app.py.")
    else:
        metric_cols = [c for c in metrics_df.columns if c != "Model"]
        model_order = list(metrics_df["Model"])

        for start in range(0, len(metric_cols), 3):
            batch = metric_cols[start:start + 3]
            tile_cols = st.columns(len(batch))
            for col, m in zip(tile_cols, batch):
                best_row = metrics_df.loc[metrics_df[m].idxmax()]
                col.metric(f"Best {m}", f"{best_row[m]:.3f}")
                col.caption(str(best_row["Model"]))

        long_df = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
        base = alt.Chart(long_df).encode(
            x=alt.X("Metric:N", sort=metric_cols, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Model:N", sort=model_order, title=None),
        )
        heat = base.mark_rect().encode(
            color=alt.Color("Value:Q", legend=None,
                            scale=alt.Scale(domain=[0, 1], range=["#e8f5f3", TEAL])),
            tooltip=["Model:N", "Metric:N", alt.Tooltip("Value:Q", format=".3f")],
        )
        labels = base.mark_text(fontSize=13, fontWeight="bold").encode(
            text=alt.Text("Value:Q", format=".3f"),
            color=alt.condition(alt.datum.Value > 0.6, alt.value("white"), alt.value("#12303a")),
        )
        st.markdown("**All metrics at a glance**")
        show_chart((heat + labels).properties(height=max(120, 55 * len(model_order))))

        chosen = st.selectbox("Compare one metric", metric_cols, index=len(metric_cols) - 1)
        one = long_df[long_df["Metric"] == chosen]
        bar = alt.Chart(one).mark_bar(size=26).encode(
            y=alt.Y("Model:N", sort="-x", title=None),
            x=alt.X("Value:Q", title=chosen, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", legend=None,
                            scale=alt.Scale(domain=model_order, range=PALETTE[: len(model_order)])),
            tooltip=["Model:N", alt.Tooltip("Value:Q", format=".3f")],
        )
        show_chart(bar.properties(height=max(120, 48 * len(model_order))))
        with st.expander("View table"):
            wide(st.dataframe, metrics_df, hide_index=True)

# ==================================================================
# THRESHOLDS (needs assets/threshold_results.csv)
# ==================================================================
with tab_threshold:
    st.subheader("Random Forest threshold study")
    st.caption("These held-out results use Random Forest. Metrics on Compare use the default 0.5 threshold.")
    study_threshold = 0.25
    thr_df = read_csv_if_exists("threshold_results.csv")
    if thr_df is None:
        st.info("threshold_results.csv not found. Put it in an `assets` folder next to app.py.")
    else:
        st.write(
            "Lowering the threshold catches more stroke-class records (higher recall) "
            "but raises false positives."
        )
        plot_cols = [c for c in ["Precision", "Recall", "F1"] if c in thr_df.columns]
        color_map = {"Precision": INDIGO, "Recall": TEAL, "F1": AMBER}
        tl = thr_df.melt(id_vars="Threshold", value_vars=plot_cols,
                         var_name="Metric", value_name="Value")
        lines = alt.Chart(tl).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("Threshold:Q"),
            y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N", scale=alt.Scale(
                domain=plot_cols, range=[color_map[c] for c in plot_cols])),
            tooltip=["Metric:N", alt.Tooltip("Threshold:Q", format=".2f"),
                     alt.Tooltip("Value:Q", format=".3f")],
        )
        rule = alt.Chart(pd.DataFrame({"Threshold": [float(study_threshold)]})).mark_rule(
            color=SLATE, strokeDash=[6, 4], strokeWidth=2
        ).encode(x="Threshold:Q")
        st.markdown("**Precision, recall and F1 by threshold**")
        show_chart((lines + rule).properties(height=300))

        if "False Positives" in thr_df.columns and "Recall" in thr_df.columns:
            tdf = thr_df.rename(columns={"False Positives": "FalsePositives"})
            base = alt.Chart(tdf).encode(x=alt.X("Threshold:Q"))
            bars = base.mark_bar(color=AMBER, opacity=0.5, size=16).encode(
                y=alt.Y("FalsePositives:Q", title="False positives (bars)"),
                tooltip=[alt.Tooltip("Threshold:Q", format=".2f"), "FalsePositives:Q"],
            )
            recall_line = base.mark_line(color=TEAL, strokeWidth=3, point=True).encode(
                y=alt.Y("Recall:Q", title="Recall (line)",
                        axis=alt.Axis(orient="right"), scale=alt.Scale(domain=[0, 1])),
                tooltip=[alt.Tooltip("Threshold:Q", format=".2f"),
                         alt.Tooltip("Recall:Q", format=".3f")],
            )
            st.markdown("**The trade-off: recall vs false positives**")
            show_chart(alt.layer(bars, recall_line).resolve_scale(y="independent").properties(height=300))

        nearest = thr_df.iloc[(thr_df["Threshold"] - float(study_threshold)).abs().argmin()]
        st.caption(
            f"Dashed grey line = RF threshold used in this study ({study_threshold}). "
            f"Closest row in the table: {nearest['Threshold']:.2f}."
        )
        with st.expander("View full table"):
            wide(st.dataframe, thr_df, hide_index=True)

# ==================================================================
# EXPLAIN (computed live from the saved model)
# ==================================================================
with tab_explain:
    st.subheader("What the model relies on")
    importances, imp_label = get_importances(model, feature_columns)
    if importances is None:
        st.info(f"The saved model (`{model_name}`) does not expose importances or coefficients.")
    else:
        if len(importances) > 3:
            top_n = st.slider("Number of features to show", 3, len(importances),
                              min(10, len(importances)))
        else:
            top_n = len(importances)
        imp_df = importances.head(top_n).reset_index()
        imp_df.columns = ["Feature", "Value"]
        imp_chart = alt.Chart(imp_df).mark_bar(color=TEAL).encode(
            x=alt.X("Value:Q", title=imp_label),
            y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=["Feature:N", alt.Tooltip("Value:Q", format=".4f")],
        )
        show_chart(imp_chart.properties(height=max(160, 32 * top_n)))
        st.caption(
            f"{imp_label} from the saved `{model_name}`. This shows which inputs "
            "the model uses most, not whether they cause stroke."
        )

# ==================================================================
# EDA (needs assets/*.png)
# ==================================================================
with tab_eda:
    st.subheader("EDA and evaluation")
    images = [
        ("class_distribution.png", "Class distribution"),
        ("age_by_stroke.png", "Age distribution by class"),
        ("confusion_matrices.png", "Confusion matrices"),
        ("roc_curves.png", "ROC curves"),
    ]
    found_any = False
    for filename, caption in images:
        found_any = show_image_if_exists(filename, caption) or found_any
    if not found_any:
        st.info("No images found. Put them in an `assets` folder next to app.py.")