"""
app.py  --  Bias-Aware Loan / Credit Risk Prediction System
============================================================
Run:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from fairness_metrics import (
    fairness_summary, compute_fairness_metrics,
    reweigh_samples, threshold_optimization,
)
from bias_pipeline import (
    run_german_pipeline, run_loan_pipeline,
    load_german, load_loan_train,
    preprocess_german, preprocess_loan,
    GERMAN_PROTECTED, LOAN_PROTECTED,
)
from eda_report import german_eda, loan_eda

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bias-Aware Prediction System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme state ───────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True   # default: dark


def _inject_theme():
    """Inject full CSS for the currently active theme."""
    dark = st.session_state.dark_mode

 

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/loan.jpg", width=70)
    st.title("⚖️ FairLoan AI")

  

    st.markdown("---")

    dataset_choice = st.selectbox(
        "📂 Select Dataset",
        ["German Credit Risk", "Loan Eligibility (Train/Test)"],
    )

    st.markdown("---")
    section = st.radio(
        "📋 Navigate",
        ["🏠 Overview", "📊 EDA & Distributions", "🤖 Model Training",
         "⚖️ Fairness & Bias", "🔧 Bias Mitigation", "🔮 Live Prediction"],
    )

    st.markdown("---")
    st.caption("All models trained on limited data sets")

IS_GERMAN = (dataset_choice == "German Credit Risk")

# ─────────────────────────────────────────────────────────────────────────────
# CACHED PIPELINE LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_german_pipeline():
    return run_german_pipeline()

@st.cache_resource(show_spinner=False)
def get_loan_pipeline():
    return run_loan_pipeline()

@st.cache_data(show_spinner=False)
def get_german_eda_cached():
    df = pd.read_csv("german_credit_risk.csv")
    return german_eda(df), df

@st.cache_data(show_spinner=False)
def get_loan_eda_cached():
    df = pd.read_csv("loan_train_cleaned.csv")
    return loan_eda(df), df

# ─────────────────────────────────────────────────────────────────────────────
# CACHED PREPROCESSING  (same encoders/scaler used for training)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_german_preprocessor():
    """Returns (feature_cols, encoders, scaler) fitted on full German dataset."""
    df = load_german()
    _, _, feature_cols, encoders, scaler, _ = preprocess_german(df)
    return feature_cols, encoders, scaler

@st.cache_resource(show_spinner=False)
def get_loan_preprocessor():
    """Returns (feature_cols, encoders, scaler) fitted on full Loan train set."""
    train_df = load_loan_train()
    _, _, feature_cols, encoders, scaler, _, _ = preprocess_loan(train_df)
    return feature_cols, encoders, scaler

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _chart_layout(**extra):
    """Return a Plotly layout dict styled for the current theme."""
    dark = st.session_state.get("dark_mode", True)
    bg   = "#1c2333" if dark else "#ffffff"
    text = "#e6edf3" if dark else "#1a1a2e"
    grid = "#30363d" if dark else "#e0e0e0"
    return dict(
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color=text),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid),
        legend=dict(bgcolor=bg, font=dict(color=text)),
        **extra,
    )


def perf_gauge(value, title):
    dark  = st.session_state.get("dark_mode", True)
    bg    = "#1c2333" if dark else "#ffffff"
    text  = "#e6edf3" if dark else "#1a1a2e"
    steps_dark  = [
        {"range": [0, 60],   "color": "#3b1a1a"},
        {"range": [60, 80],  "color": "#3b3200"},
        {"range": [80, 100], "color": "#0d2b1a"},
    ]
    steps_light = [
        {"range": [0, 60],   "color": "#fee2e2"},
        {"range": [60, 80],  "color": "#fef3c7"},
        {"range": [80, 100], "color": "#d1fae5"},
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title, "font": {"size": 14, "color": text}},
        number={"suffix": "%", "font": {"color": text}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": text},
            "bar": {"color": "#667eea"},
            "bgcolor": bg,
            "steps": steps_dark if dark else steps_light,
            "threshold": {"line": {"color": "red", "width": 3},
                          "thickness": 0.75, "value": 80},
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color=text),
    )
    return fig



def render_fairness_verdicts(fairness_df):
    """Render coloured verdict blocks using the fairness_summary output."""
    verdicts = fairness_summary(fairness_df)
    for label, detail in verdicts:
        icon   = "✅" if label.startswith("PASS") else "⚠️"
        css    = "bias-ok" if label.startswith("PASS") else "bias-warning"
        pretty = label.replace("PASS ", "").replace("WARN ", "")
        st.markdown(
            f'<div class="{css}"><b>{icon} {pretty}</b>: {detail}</div>',
            unsafe_allow_html=True,
        )


def safe_predict(model, X):
    """Predict class + probabilities with graceful fallback."""
    pred = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        proba = None
    return pred, proba


def transform_german_input(row_dict):
    """Encode and scale a single German Credit input row."""
    feature_cols, encoders, scaler = get_german_preprocessor()
    df_row = pd.DataFrame([row_dict])
    errors = []

    # Validate column presence
    for col in feature_cols:
        if col not in df_row.columns:
            errors.append(f"Missing feature: {col}")

    if errors:
        return None, errors

    for col, le in encoders.items():
        if col in df_row.columns:
            val = str(df_row[col].iloc[0])
            if val not in le.classes_:
                errors.append(f"Unknown value '{val}' for feature '{col}'. "
                              f"Expected one of: {list(le.classes_)}")
            else:
                df_row[col] = le.transform([val])

    if errors:
        return None, errors

    X = scaler.transform(df_row[feature_cols])
    return X, []


def transform_loan_input(row_dict):
    """Encode and scale a single Loan input row."""
    feature_cols, encoders, scaler = get_loan_preprocessor()
    df_row = pd.DataFrame([row_dict])
    errors = []

    for col in feature_cols:
        if col not in df_row.columns:
            errors.append(f"Missing feature: {col}")

    if errors:
        return None, errors

    for col, le in encoders.items():
        if col in df_row.columns:
            val = str(df_row[col].iloc[0])
            if val not in le.classes_:
                # Use nearest class rather than crash (still warn)
                df_row[col] = 0
            else:
                df_row[col] = le.transform([val])

    if errors:
        return None, errors

    X = scaler.transform(df_row[feature_cols])
    return X, []


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: 🏠 OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if section == "🏠 Overview":
    st.title("⚖️ Bias-Aware Prediction System")
    st.markdown("""
    > **A production-grade system** combining ML model training, fairness auditing,
    > bias detection, and mitigation across two distinct real-world datasets.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📁 Datasets", "2")
    c2.metric("🤖 ML Models", "3")
    c3.metric("⚖️ Fairness Metrics", "5")
    c4.metric("🔧 Mitigations", "2")

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 📦 German Credit Risk Dataset")
        st.markdown("""
        - **1,000 samples**, 10 features
        - **Target:** `Risk` → **Good Risk** / **Bad Risk**
        - **Meaning:** creditworthiness classification
        - **Protected:** Sex, Age
        - Features: Job, Housing, Credit Amount, Duration, Purpose…
        """)
    with col_r:
        st.markdown("### 📦 Loan Eligibility Dataset")
        st.markdown("""
        - **614 train / 367 test** samples, 12 features
        - **Target:** `Loan_Status` → **Eligible (Y)** / **Not Eligible (N)**
        - **Meaning:** loan approval decision
        - **Protected:** Gender, Married, Education
        - Features: Income, Loan Amount, Credit History, Property Area…
        """)

    st.markdown("---")
    st.markdown("### 🗺️ System Architecture")
    st.code("""
Raw CSV Data
    ↓
Preprocessing + Encoding + Scaling
    ↓
EDA & Bias Detection  (Disparate Impact Analysis)
    ↓
Model Training  (Logistic Regression | Random Forest | XGBoost)
    ↓
Fairness Audit  (Selection Rate | TPR | FPR | FNR | DI | Dem. Parity | Eq. Odds)
    ↓
Bias Mitigation  (Reweighing | Threshold Optimization)
    ↓
Post-Mitigation Re-evaluation
    ↓
Live Prediction Interface  (per-dataset fields, labels, target mapping)
""", language="text")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: 📊 EDA
# ─────────────────────────────────────────────────────────────────────────────
elif section == "📊 EDA & Distributions":
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    if IS_GERMAN:
        with st.spinner("Generating EDA charts…"):
            plots, df = get_german_eda_cached()
        st.subheader("German Credit Risk Dataset")
        tab1, tab2, tab3 = st.tabs(
            ["📈 Credit Risk & Demographics", "💰 Credit Features", "🔗 Correlations"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                if os.path.exists(plots["german_target"]):
                    st.image(plots["german_target"], caption="Credit Risk Distribution")
            with c2:
                if os.path.exists(plots["german_sex_approval"]):
                    st.image(plots["german_sex_approval"],
                             caption="Good-Risk Rate by Sex")
            if os.path.exists(plots["german_age"]):
                st.image(plots["german_age"], caption="Age Distribution by Risk",
                         use_container_width=True)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                if os.path.exists(plots["german_credit_purpose"]):
                    st.image(plots["german_credit_purpose"],
                             caption="Credit Amount by Purpose")
            with c2:
                if os.path.exists(plots["german_job_risk"]):
                    st.image(plots["german_job_risk"], caption="Job Level vs Risk")

        with tab3:
            if os.path.exists(plots["german_corr"]):
                st.image(plots["german_corr"], caption="Feature Correlation Heatmap",
                         use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Dataset Statistics")
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.dataframe(df.describe(), use_container_width=True)
        with col_stat2:
            fig = px.pie(df, names="Risk", title="Credit Risk Distribution",
                         color_discrete_map={"good": "#2ecc71", "bad": "#e74c3c"})
            fig.update_layout(**_chart_layout())
            st.plotly_chart(fig, use_container_width=True)

    else:
        with st.spinner("Generating EDA charts…"):
            plots, df = get_loan_eda_cached()
        st.subheader("Loan Eligibility Dataset")
        tab1, tab2, tab3 = st.tabs(
            ["📈 Loan Status & Demographics", "💰 Loan Features", "🔗 Correlations"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                if os.path.exists(plots["loan_target"]):
                    st.image(plots["loan_target"], caption="Loan Status Distribution")
            with c2:
                if os.path.exists(plots["loan_gender_approval"]):
                    st.image(plots["loan_gender_approval"],
                             caption="Approval Rate by Gender")
            c3, c4 = st.columns(2)
            with c3:
                if os.path.exists(plots["loan_education"]):
                    st.image(plots["loan_education"],
                             caption="Approval Rate by Education")
            with c4:
                if os.path.exists(plots["loan_area"]):
                    st.image(plots["loan_area"],
                             caption="Approval Rate by Property Area")

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                if os.path.exists(plots["loan_income"]):
                    st.image(plots["loan_income"],
                             caption="Applicant Income Distribution")
            with c2:
                if os.path.exists(plots["loan_amount_status"]):
                    st.image(plots["loan_amount_status"],
                             caption="Loan Amount by Approval Status")
            if os.path.exists(plots["loan_credit_hist"]):
                st.image(plots["loan_credit_hist"],
                         caption="Credit History vs Loan Approval",
                         use_container_width=True)

        with tab3:
            if os.path.exists(plots["loan_corr"]):
                st.image(plots["loan_corr"], caption="Feature Correlation Heatmap",
                         use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Dataset Statistics")
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.dataframe(df.describe(), use_container_width=True)
        with col_stat2:
            fig = px.pie(df, names="Loan_Status", title="Loan Status Distribution",
                         color_discrete_map={"Y": "#2ecc71", "N": "#e74c3c"})
            fig.update_layout(**_chart_layout())
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: 🤖 MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
elif section == "🤖 Model Training":
    st.markdown('<div class="section-header">🤖 Model Training & Performance</div>',
                unsafe_allow_html=True)

    with st.spinner("Training models… first run may take 30–60 s."):
        pipeline = get_german_pipeline() if IS_GERMAN else get_loan_pipeline()

    st.success(f"Models trained on **{pipeline['dataset']}**")

    model_names  = list(pipeline["base"].keys())
    chosen_model = st.selectbox("Select Model", model_names)
    base         = pipeline["base"][chosen_model]
    perf         = base["performance"]

    # Target semantics caption
    if IS_GERMAN:
        st.caption("Target: **Risk** (bad=0 → Bad Risk | good=1 → Good Risk)")
    else:
        st.caption("Target: **Loan_Status** (N=0 → Not Eligible | Y=1 → Eligible)")

    st.markdown(f"### {chosen_model} — Performance Metrics")

    g1, g2, g3, g4 = st.columns(4)
    with g1: st.plotly_chart(perf_gauge(perf["Accuracy"],  "Accuracy"),  use_container_width=True)
    with g2: st.plotly_chart(perf_gauge(perf["F1 Score"],  "F1 Score"),  use_container_width=True)
    with g3: st.plotly_chart(perf_gauge(perf["Precision"], "Precision"), use_container_width=True)
    with g4:
        auc = perf["AUC-ROC"] or 0.0
        st.plotly_chart(perf_gauge(auc, "AUC-ROC"), use_container_width=True)

    # Confusion matrix with dataset-correct axis labels
    st.markdown("### 🧩 Confusion Matrix")
    cm = np.array(perf["Confusion Matrix"])
    if IS_GERMAN:
        axis_labels = ["Bad Risk (0)", "Good Risk (1)"]
    else:
        axis_labels = ["Not Eligible (0)", "Eligible (1)"]

    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual"},
        x=axis_labels, y=axis_labels,
        title=f"Confusion Matrix — {chosen_model}",
    )
    fig_cm.update_layout(**_chart_layout())
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("### 📋 Classification Report")
    cr_df = pd.DataFrame(perf["Classification Report"]).T.round(3)
    st.dataframe(cr_df.style.background_gradient(cmap="YlGn"), use_container_width=True)

    st.markdown("### 🏆 All Models Comparison")
    rows = []
    for mn in model_names:
        p = pipeline["base"][mn]["performance"]
        rows.append({"Model": mn, "Accuracy": p["Accuracy"],
                     "F1": p["F1 Score"], "Precision": p["Precision"],
                     "Recall": p["Recall"], "AUC-ROC": p["AUC-ROC"]})
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df.style.highlight_max(axis=0, color="#d4edda"),
                 use_container_width=True)

    fig_bar = px.bar(
        comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Metric", y="Score", color="Model", barmode="group",
        title="Model Comparison", range_y=[0, 1],
    )
    fig_bar.update_layout(**_chart_layout())
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: ⚖️ FAIRNESS & BIAS
# ─────────────────────────────────────────────────────────────────────────────
elif section == "⚖️ Fairness & Bias":
    st.markdown('<div class="section-header">⚖️ Fairness Audit & Bias Detection</div>',
                unsafe_allow_html=True)

    with st.spinner("Running fairness audit…"):
        if IS_GERMAN:
            pipeline = get_german_pipeline()
            sensitive = "Sex"               # fix #9 — German uses Sex
            rate_label = "Good-Risk Rate"
        else:
            pipeline = get_loan_pipeline()
            sensitive = "Gender"            # fix #9 — Loan uses Gender
            rate_label = "Loan Approval Rate"

    model_names  = list(pipeline["base"].keys())
    chosen_model = st.selectbox("Select Model", model_names)

    # Toggle: base vs mitigated
    use_mit = st.checkbox("Compare with bias-mitigated model", value=False)

    base      = pipeline["base"][chosen_model]
    fair_data = pipeline["fair"][chosen_model]

    fairness_df      = base["fairness"]
    fairness_df_fair = fair_data["fairness"]

    st.markdown(f"### Fairness Metrics — {chosen_model}")
    st.markdown(f"**Sensitive attribute:** `{sensitive}`")

    # ── Verdicts ──────────────────────────────────────────────────────────
    col_v1, col_v2 = (st.columns(2) if use_mit else (st.container(), None))
    with col_v1:
        st.markdown("#### Baseline Model — Fairness Verdicts")
        render_fairness_verdicts(fairness_df)

    if use_mit and col_v2:
        with col_v2:
            st.markdown("#### Bias-Mitigated Model — Fairness Verdicts")
            render_fairness_verdicts(fairness_df_fair)

    st.markdown("---")

    # ── Per-group metrics table with Selection Rate, TPR, FPR, FNR ───────
    st.markdown("#### Per-Group Metrics (Baseline)")
    display_cols = ["Group", "N", "Selection Rate (PPR)",
                    "TPR (Recall)", "FPR", "FNR", "Accuracy"]
    st.dataframe(
        fairness_df[display_cols].style.background_gradient(
            cmap="RdYlGn",
            subset=["Selection Rate (PPR)", "TPR (Recall)", "Accuracy"],
        ),
        use_container_width=True,
    )
    st.caption("FNR = False Negative Rate (miss rate = 1 - TPR). "
               "FPR = False Positive Rate. PPR = Selection / Approval Rate.")

    if use_mit:
        st.markdown("#### Per-Group Metrics (Bias-Mitigated)")
        st.dataframe(
            fairness_df_fair[display_cols].style.background_gradient(
                cmap="RdYlGn",
                subset=["Selection Rate (PPR)", "TPR (Recall)", "Accuracy"],
            ),
            use_container_width=True,
        )

    # ── Selection Rate bar chart ──────────────────────────────────────────
    fig = px.bar(
        fairness_df, x="Group", y="Selection Rate (PPR)",
        title=f"{rate_label} by Group — Baseline",
        color="Selection Rate (PPR)",
        color_continuous_scale="RdYlGn",
        range_y=[0, 1],
    )
    fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                  annotation_text="80% DI Threshold")
    fig.update_layout(**_chart_layout())
    st.plotly_chart(fig, use_container_width=True)

    # ── TPR / FPR / FNR ──────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        fig_tpr = px.bar(fairness_df, x="Group", y="TPR (Recall)",
                         title="True Positive Rate (TPR)", color="Group")
        fig_tpr.update_layout(**_chart_layout())
        st.plotly_chart(fig_tpr, use_container_width=True)
    with c2:
        fig_fpr = px.bar(fairness_df, x="Group", y="FPR",
                         title="False Positive Rate (FPR)", color="Group")
        fig_fpr.update_layout(**_chart_layout())
        st.plotly_chart(fig_fpr, use_container_width=True)
    with c3:
        fig_fnr = px.bar(fairness_df, x="Group", y="FNR",
                         title="False Negative Rate (FNR)", color="Group")
        fig_fnr.update_layout(**_chart_layout())
        st.plotly_chart(fig_fnr, use_container_width=True)

    # ── Summary badges ────────────────────────────────────────────────────
    st.markdown("#### Summary Fairness Scores")
    di  = fairness_df["Disparate Impact"].iloc[0]
    dpd = fairness_df["Dem. Parity Diff"].iloc[0]
    eod = fairness_df["Eq. Odds Diff (TPR)"].iloc[0]

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Disparate Impact", f"{di:.3f}",
                  delta=">=0.80 = FAIR",
                  delta_color="normal" if di >= 0.8 else "inverse")
    with m2:
        st.metric("Dem. Parity Diff", f"{dpd:.3f}",
                  delta="<=0.10 = FAIR",
                  delta_color="normal" if dpd <= 0.1 else "inverse")
    with m3:
        st.metric("Eq. Odds Diff (TPR)", f"{eod:.3f}",
                  delta="<=0.10 = FAIR",
                  delta_color="normal" if eod <= 0.1 else "inverse")

    # ── Optimised thresholds ──────────────────────────────────────────────
    if base["opt_thresholds"]:
        st.markdown("---")
        st.markdown("#### Per-Group Optimised Decision Thresholds")
        thresh_df = pd.DataFrame(
            list(base["opt_thresholds"].items()), columns=["Group", "Threshold"])
        st.dataframe(thresh_df, use_container_width=True)
        st.caption("Thresholds optimised per-group using Youden's J statistic (TPR - FPR).")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: 🔧 BIAS MITIGATION
# ─────────────────────────────────────────────────────────────────────────────
elif section == "🔧 Bias Mitigation":
    st.markdown('<div class="section-header">🔧 Bias Mitigation — Reweighing</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Technique:** Sample Reweighing — assigns higher weights to under-represented
    (group, label) combinations so the model sees balanced evidence during training.

    > **Note:** Bias mitigation reduces group disparity but does **not** guarantee a
    > perfectly fair prediction for every individual. Always interpret results with context.
    """)

    with st.spinner("Loading mitigated models…"):
        pipeline = get_german_pipeline() if IS_GERMAN else get_loan_pipeline()

    if IS_GERMAN:
        rate_col  = "Selection Rate (PPR)"
        rate_title = "Good-Risk Rate by Group"
    else:
        rate_col  = "Selection Rate (PPR)"
        rate_title = "Loan Approval Rate by Group"

    model_names  = list(pipeline["base"].keys())
    chosen_model = st.selectbox("Select Model", model_names)

    base_perf = pipeline["base"][chosen_model]["performance"]
    fair_perf = pipeline["fair"][chosen_model]["performance"]
    base_fair = pipeline["base"][chosen_model]["fairness"]
    fair_fair = pipeline["fair"][chosen_model]["fairness"]

    st.markdown(f"### Before vs After Mitigation — {chosen_model}")

    # Performance bar comparison
    metrics      = ["Accuracy", "F1 Score", "Precision", "Recall", "AUC-ROC"]
    before_vals  = [base_perf[m] or 0 for m in metrics]
    after_vals   = [fair_perf[m] or 0 for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Baseline Model",      x=metrics, y=before_vals,
                         marker_color="#e74c3c"))
    fig.add_trace(go.Bar(name="Bias-Mitigation Model", x=metrics, y=after_vals,
                         marker_color="#2ecc71"))
    fig.update_layout(barmode="group",
                      title="Performance: Baseline vs Bias-Mitigation Model",
                      yaxis_range=[0, 1])
    fig.update_layout(**_chart_layout())
    st.plotly_chart(fig, use_container_width=True)

    # Fairness delta metrics
    st.markdown("### Fairness Change: Baseline vs Bias-Mitigation Model")

    def _delta_metric(before, after, label, lower_is_better=False):
        delta = round(after - before, 4)
        if lower_is_better:
            good = delta <= 0
        else:
            good = delta >= 0
        st.metric(label, f"{after:.3f}",
                  delta=f"{delta:+.3f} vs {before:.3f}",
                  delta_color="normal" if good else "inverse")

    c1, c2, c3 = st.columns(3)
    with c1:
        _delta_metric(base_fair["Disparate Impact"].iloc[0],
                      fair_fair["Disparate Impact"].iloc[0],
                      "Disparate Impact", lower_is_better=False)
    with c2:
        _delta_metric(base_fair["Dem. Parity Diff"].iloc[0],
                      fair_fair["Dem. Parity Diff"].iloc[0],
                      "Dem. Parity Diff", lower_is_better=True)
    with c3:
        _delta_metric(base_fair["Eq. Odds Diff (TPR)"].iloc[0],
                      fair_fair["Eq. Odds Diff (TPR)"].iloc[0],
                      "Eq. Odds Diff (TPR)", lower_is_better=True)

    # Approval rate comparison
    all_groups = base_fair["Group"].tolist()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Baseline Model",
                          x=all_groups, y=base_fair[rate_col].tolist(),
                          marker_color="#e74c3c"))
    fig2.add_trace(go.Bar(name="Bias-Mitigation Model",
                          x=all_groups, y=fair_fair[rate_col].tolist(),
                          marker_color="#2ecc71"))
    fig2.add_hline(y=0.8, line_dash="dash", line_color="orange",
                   annotation_text="80% DI Threshold")
    fig2.update_layout(barmode="group",
                       title=f"{rate_title} — Before vs After",
                       yaxis_range=[0, 1])
    fig2.update_layout(**_chart_layout())
    st.plotly_chart(fig2, use_container_width=True)

    # Detailed comparison table
    st.markdown("### Detailed Comparison Table")
    merged = base_fair[["Group", rate_col, "TPR (Recall)", "FPR", "FNR", "Accuracy"]].copy()
    merged.columns = ["Group", "Sel.Rate (Before)", "TPR (Before)",
                      "FPR (Before)", "FNR (Before)", "Acc (Before)"]
    for col_after, col_src in [
        ("Sel.Rate (After)", rate_col), ("TPR (After)", "TPR (Recall)"),
        ("FPR (After)", "FPR"), ("FNR (After)", "FNR"), ("Acc (After)", "Accuracy"),
    ]:
        merged[col_after] = fair_fair[col_src].values
    st.dataframe(merged.round(4), use_container_width=True)

    st.markdown('<div class="mitigation-note">After reweighing, selection rates across groups '
                'converge toward each other, increasing the Disparate Impact score toward '
                'the 0.80 fairness threshold. A small accuracy trade-off is expected and '
                'acceptable.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: 🔮 LIVE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
elif section == "🔮 Live Prediction":

    # ── Dataset-specific titles / semantics  (fix #1, #10) ───────────────
    if IS_GERMAN:
        page_title    = "🏦 German Credit Risk — Enter Applicant Details"
        btn_label     = "🔮 Predict Credit Risk"
        result_label  = "Credit Risk"
        pos_class     = "Good Risk"   # label for pred == 1   (fix #5: good=1)
        neg_class     = "Bad Risk"    # label for pred == 0
        pos_prob_lbl  = "Good Risk Probability"
        neg_prob_lbl  = "Bad Risk Probability"
    else:
        page_title    = "🏦 Loan Eligibility — Enter Applicant Details"
        btn_label     = "🔮 Predict Loan Eligibility"
        result_label  = "Loan Status"
        pos_class     = "Eligible"
        neg_class     = "Not Eligible"
        pos_prob_lbl  = "Approval Probability"
        neg_prob_lbl  = "Rejection Probability"

    st.markdown(f'<div class="section-header">{page_title}</div>',
                unsafe_allow_html=True)

    with st.spinner("Loading trained models…"):
        pipeline = get_german_pipeline() if IS_GERMAN else get_loan_pipeline()

    model_names = list(pipeline["base"].keys())

    # Bias-mitigation toggle  (fix #8)
    use_fair = st.checkbox("Use bias-mitigated (reweighed) model", value=True)
    model_pool = pipeline["fair"] if use_fair else pipeline["base"]

    if use_fair:
        st.markdown(
            '<div class="mitigation-note">⚠️ <b>Bias-Mitigation Model</b> active. '
            'Reweighing reduces group disparity but does <b>not</b> guarantee a fair '
            'outcome for every individual.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Input form ────────────────────────────────────────────────────────
    if IS_GERMAN:
        # fix #2: only German fields shown here
        st.subheader("Applicant Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            age      = st.slider("Age", 18, 75, 30)
            sex      = st.selectbox("Sex", ["male", "female"])
            job      = st.selectbox("Job Level", [0, 1, 2, 3],
                                    help="0=Unskilled non-resident, 1=Unskilled resident, "
                                         "2=Skilled, 3=Highly skilled")
        with col2:
            housing  = st.selectbox("Housing", ["own", "free", "rent"])
            saving   = st.selectbox("Saving Accounts",
                                    ["little", "moderate", "quite rich", "rich"])
            checking = st.selectbox("Checking Account",
                                    ["little", "moderate", "rich"])
        with col3:
            credit_amt = st.number_input("Credit Amount (DM)", 250, 20000, 3000, step=50)
            duration   = st.slider("Duration (months)", 4, 72, 24)
            purpose    = st.selectbox("Purpose", [
                "car", "furniture/equipment", "radio/tv",
                "domestic appliance", "repairs",
                "education", "business", "vacation/others",
            ])

        input_row = {
            "Age": age, "Sex": sex, "Job": job, "Housing": housing,
            "Saving_accounts": saving, "Checking_account": checking,
            "Credit_amount": credit_amt, "Duration": duration, "Purpose": purpose,
        }
        transform_fn = transform_german_input

    else:
        # fix #3: only Loan fields shown here
        st.subheader("Applicant Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender       = st.selectbox("Gender", ["Male", "Female"])
            married      = st.selectbox("Married", ["Yes", "No"])
            dependents   = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        with col2:
            education    = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_emp     = st.selectbox("Self Employed", ["Yes", "No"])
            prop_area    = st.selectbox("Property Area",
                                        ["Urban", "Semiurban", "Rural"])
        with col3:
            app_income   = st.number_input("Applicant Income", 150, 100000, 5000, step=100)
            coapp_income = st.number_input("Coapplicant Income", 0, 50000, 0, step=100)
            loan_amt     = st.number_input("Loan Amount (thousands)", 10, 700, 150, step=5)
            loan_term    = st.selectbox("Loan Term (months)",
                                        [360, 180, 120, 60, 480, 84, 300, 240, 36, 12])
            credit_hist  = st.selectbox("Credit History",
                                        [1.0, 0.0],
                                        format_func=lambda v: "Good (1)" if v == 1.0 else "Bad (0)")

        input_row = {
            "Gender": gender, "Married": married, "Dependents": dependents,
            "Education": education, "Self_Employed": self_emp,
            "ApplicantIncome": int(app_income),
            "CoapplicantIncome": float(coapp_income),
            "LoanAmount": float(loan_amt),
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": float(credit_hist),
            "Property_Area": prop_area,
        }
        transform_fn = transform_loan_input

    st.markdown("---")

    # ── Predict button ────────────────────────────────────────────────────
    if st.button(btn_label, type="primary"):

        # fix #11: pre-flight checks
        X_input, errors = transform_fn(input_row)

        if errors:
            st.error("**Input validation failed:**\n" + "\n".join(f"- {e}" for e in errors))
            st.stop()

        # ── fix #7: all-model comparison table ───────────────────────────
        st.markdown(f"### {result_label} — All Models Comparison")

        comparison_rows = []
        prob_data_all   = {}   # model_name -> proba array

        for mn in model_names:
            m_obj = model_pool[mn]["model"]
            # fix #4: use same preprocessor as training (already cached above)
            pred, proba = safe_predict(m_obj, X_input)

            # fix #5: German target mapping  bad=0 → Bad Risk, good=1 → Good Risk
            label = pos_class if pred == 1 else neg_class
            conf  = (f"{proba[pred]*100:.1f}%" if proba is not None
                     else "N/A — no predict_proba")
            mitigation_tag = "Bias-Mitigated" if use_fair else "Baseline"
            comparison_rows.append({
                "Model":       mn,
                "Type":        mitigation_tag,
                f"{result_label}": label,
                "Confidence":  conf,
            })
            prob_data_all[mn] = proba

        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── fix #6: individual model deep-dive ───────────────────────────
        selected_for_detail = st.selectbox(
            "Show probability detail for:", model_names, key="detail_model")

        m_obj_detail = model_pool[selected_for_detail]["model"]
        pred_d, proba_d = safe_predict(m_obj_detail, X_input)
        pred_label = pos_class if pred_d == 1 else neg_class

        # Styled result banner
        box_css = "pred-box-good" if pred_d == 1 else "pred-box-bad"
        icon    = "✅" if pred_d == 1 else "❌"
        st.markdown(
            f'<div class="{box_css}">'
            f'{icon} <b>{result_label}:</b> {pred_label}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if proba_d is not None:
            # fix #6: show both class probabilities
            st.markdown("**Prediction Probabilities:**")
            pm1, pm2 = st.columns(2)
            with pm1:
                st.metric(pos_prob_lbl, f"{proba_d[1]*100:.1f}%")
            with pm2:
                st.metric(neg_prob_lbl, f"{proba_d[0]*100:.1f}%")

            fig_prob = go.Figure(go.Bar(
                x=[neg_class, pos_class],
                y=[proba_d[0], proba_d[1]],
                marker_color=["#e74c3c", "#2ecc71"],
                text=[f"{proba_d[0]*100:.1f}%", f"{proba_d[1]*100:.1f}%"],
                textposition="outside",
            ))
            fig_prob.update_layout(
                title=f"Prediction Probability — {selected_for_detail}",
                yaxis_range=[0, 1.15],
                yaxis_title="Probability",
            )
            fig_prob.update_layout(**_chart_layout())
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("This model does not expose predict_proba. "
                    "Calibration is required to obtain probabilities.")

        # ── Probability comparison across all models ──────────────────────
        if all(v is not None for v in prob_data_all.values()):
            st.markdown("---")
            st.markdown("#### Approval Probability Across All Models")
            model_list  = list(prob_data_all.keys())
            pos_probs   = [prob_data_all[mn][1] for mn in model_list]
            fig_all = go.Figure(go.Bar(
                x=model_list, y=pos_probs,
                marker_color=["#667eea"] * len(model_list),
                text=[f"{p*100:.1f}%" for p in pos_probs],
                textposition="outside",
            ))
            fig_all.update_layout(
                title=f"P({pos_class}) by Model",
                yaxis_range=[0, 1.15],
                yaxis_title="Probability",
            )
            fig_all.update_layout(**_chart_layout())
            st.plotly_chart(fig_all, use_container_width=True)
