"""
Diabetes Risk Prediction System
Supervised Learning Web Application built with Streamlit & Scikit-Learn
"""

import os
import io
import time
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from predictor import DiabetesPredictor

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="DiaRisk AI — Clinical Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek CSS Styling
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(120deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 16px;
        font-weight: 400;
    }
    
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #38bdf8;
        margin-right: 8px;
        margin-bottom: 6px;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(19, 27, 46, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 22px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }

    /* Risk Score Cards */
    .risk-banner-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 14px;
        padding: 20px;
    }
    
    .risk-banner-med {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 14px;
        padding: 20px;
    }
    
    .risk-banner-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 14px;
        padding: 20px;
    }

    /* Clinical Rec Cards */
    .rec-box {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #38bdf8;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    .rec-box-warning {
        border-left-color: #f59e0b;
    }
    
    .rec-box-danger {
        border-left-color: #ef4444;
    }
    
    .rec-box-success {
        border-left-color: #10b981;
    }

    /* Tab navigation polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    /* Presets row */
    .preset-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Predictor Engine with Cached Resource
@st.cache_resource
def get_predictor():
    return DiabetesPredictor()

@st.cache_data
def get_dataset():
    if os.path.exists("diabetes.csv"):
        return pd.read_csv("diabetes.csv")
    return None

predictor = get_predictor()
raw_df = get_dataset()

# Top Hero Section
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🩺 DiaRisk AI — Diabetes Risk Prediction System</div>
    <div class="hero-subtitle">
        Clinical Decision Support powered by Supervised Machine Learning Algorithms (Pima Indians Clinical Cohort)
    </div>
    <div>
        <span class="badge-pill">⚡ Supervised Classification</span>
        <span class="badge-pill">🌲 Random Forest & Gradient Boosting</span>
        <span class="badge-pill">📈 Cross-Validated ROC-AUC: 82.3%</span>
        <span class="badge-pill">🛡️ Explainable Biomarkers</span>
        <span class="badge-pill">👥 Batch Patient Screening</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar with System Status & Guidance
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1579684385127-1ef15d508118?w=500&auto=format&fit=crop&q=80", use_container_width=True)
    st.markdown("### 🧬 Clinical Parameters Reference")
    st.markdown("""
    - **Glucose**: Fasting/2-hr Plasma Glucose (mg/dL)  
      *Normal: <100 | Pre-diabetes: 100-125 | Diabetes: ≥126*
    - **BMI**: Body Mass Index (kg/m²)  
      *Normal: 18.5-24.9 | Overweight: 25-29.9 | Obese: ≥30*
    - **Blood Pressure**: Diastolic (mm Hg)  
      *Normal: <80 | Stage 1: 80-89 | Stage 2: ≥90*
    - **Insulin**: 2-Hour Serum Insulin (µU/mL)  
      *Normal fasting: 16-166 µU/mL*
    - **DPF**: Diabetes Pedigree Function (Genetic score)
    """)
    st.markdown("---")
    st.markdown("### ⚙️ Engine Information")
    st.markdown(f"**Models Available**: {len(predictor.get_model_names())}")
    st.markdown("**Dataset Sample Size**: 768 Records")
    st.markdown("**Validation Strategy**: 5-Fold Stratified CV")
    st.markdown("---")
    st.caption("⚕️ *Notice: DiaRisk AI is a clinical diagnostic aid for educational and risk-stratification purposes. Confirmatory lab tests (HbA1c / OGTT) are required for definitive diagnosis.*")

# Navigation Tabs
tabs = st.tabs([
    "🩺 Patient Risk Assessment",
    "📊 Model Performance & Benchmarks",
    "🧠 Feature Importance & Explainability",
    "🔍 Exploratory Data Analysis (EDA)",
    "📁 Batch Patient Screening"
])

# ==========================================
# TAB 1: PATIENT RISK ASSESSMENT
# ==========================================
with tabs[0]:
    st.markdown("### 📋 Enter Patient Clinical Biomarkers")
    st.markdown("Select a quick profile preset or manually adjust parameters to assess diabetes risk in real-time.")

    # Preset Patient Profiles
    col_pre1, col_pre2, col_pre3, col_pre4 = st.columns(4)
    
    # Session state for inputs
    if "p_pregnancies" not in st.session_state:
        st.session_state.p_pregnancies = 2
        st.session_state.p_glucose = 110.0
        st.session_state.p_bp = 70.0
        st.session_state.p_skin = 22.0
        st.session_state.p_insulin = 80.0
        st.session_state.p_bmi = 24.5
        st.session_state.p_dpf = 0.350
        st.session_state.p_age = 32

    def set_preset(preg, gluc, bp, skin, ins, bmi, dpf, age):
        st.session_state.p_pregnancies = preg
        st.session_state.p_glucose = float(gluc)
        st.session_state.p_bp = float(bp)
        st.session_state.p_skin = float(skin)
        st.session_state.p_insulin = float(ins)
        st.session_state.p_bmi = float(bmi)
        st.session_state.p_dpf = float(dpf)
        st.session_state.p_age = int(age)

    with col_pre1:
        if st.button("🟢 Healthy Adult Profile", use_container_width=True):
            set_preset(1, 88.0, 68.0, 20.0, 75.0, 22.4, 0.220, 28)
            st.rerun()

    with col_pre2:
        if st.button("🟡 Borderline / Pre-Diabetic", use_container_width=True):
            set_preset(3, 126.0, 78.0, 28.0, 110.0, 28.5, 0.450, 44)
            st.rerun()

    with col_pre3:
        if st.button("🔴 High-Risk Diabetic Profile", use_container_width=True):
            set_preset(6, 172.0, 88.0, 36.0, 220.0, 36.8, 0.850, 52)
            st.rerun()

    with col_pre4:
        if st.button("🎲 Random Test Patient", use_container_width=True):
            if os.path.exists("models/test_sample.csv"):
                sample_df = pd.read_csv("models/test_sample.csv")
                rand_row = sample_df.sample(1).iloc[0]
                set_preset(
                    rand_row["Pregnancies"], rand_row["Glucose"], rand_row["BloodPressure"],
                    rand_row["SkinThickness"], rand_row["Insulin"], rand_row["BMI"],
                    rand_row["DiabetesPedigreeFunction"], rand_row["Age"]
                )
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form Layout
    col_input1, col_input2, col_input3 = st.columns(3)

    with col_input1:
        st.markdown("#### 🧪 Metabolic Profile")
        p_glucose = st.slider(
            "Plasma Glucose (2-hr / Fasting)",
            min_value=50.0, max_value=250.0,
            value=float(st.session_state.p_glucose),
            step=1.0,
            help="Oral glucose tolerance test / plasma glucose in mg/dL"
        )
        p_insulin = st.slider(
            "2-Hour Serum Insulin (µU/mL)",
            min_value=0.0, max_value=600.0,
            value=float(st.session_state.p_insulin),
            step=5.0,
            help="Serum insulin concentration. (If 0, system imputes median based on population)."
        )
        p_bmi = st.slider(
            "Body Mass Index (BMI in kg/m²)",
            min_value=15.0, max_value=60.0,
            value=float(st.session_state.p_bmi),
            step=0.1,
            help="Weight in kg / (Height in meters)^2"
        )

    with col_input2:
        st.markdown("#### 🩺 Vitals & Genetics")
        p_bp = st.slider(
            "Diastolic Blood Pressure (mm Hg)",
            min_value=40.0, max_value=130.0,
            value=float(st.session_state.p_bp),
            step=1.0,
            help="Resting diastolic blood pressure"
        )
        p_skin = st.slider(
            "Triceps Skinfold Thickness (mm)",
            min_value=0.0, max_value=70.0,
            value=float(st.session_state.p_skin),
            step=1.0,
            help="Measurement of body fat layer on triceps"
        )
        p_dpf = st.slider(
            "Diabetes Pedigree Function (DPF)",
            min_value=0.05, max_value=2.50,
            value=float(st.session_state.p_dpf),
            step=0.01,
            help="Genetic risk score quantifying diabetes history in biological family"
        )

    with col_input3:
        st.markdown("#### 👤 Demographics & Algorithm")
        p_age = st.slider(
            "Patient Age (Years)",
            min_value=18, max_value=90,
            value=int(st.session_state.p_age),
            step=1
        )
        p_pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0, max_value=17,
            value=int(st.session_state.p_pregnancies),
            step=1,
            help="Gestational history (0 for males or no prior pregnancies)"
        )
        model_choice = st.selectbox(
            "Supervised Learning Classifier",
            options=predictor.get_model_names(),
            index=0,
            help="Select the supervised model to compute the prediction and probability."
        )

    # Save inputs back to session state
    st.session_state.p_glucose = p_glucose
    st.session_state.p_insulin = p_insulin
    st.session_state.p_bmi = p_bmi
    st.session_state.p_bp = p_bp
    st.session_state.p_skin = p_skin
    st.session_state.p_dpf = p_dpf
    st.session_state.p_age = p_age
    st.session_state.p_pregnancies = p_pregnancies

    # Assemble Patient Record
    patient_data = {
        "Pregnancies": p_pregnancies,
        "Glucose": p_glucose,
        "BloodPressure": p_bp,
        "SkinThickness": p_skin,
        "Insulin": p_insulin,
        "BMI": p_bmi,
        "DiabetesPedigreeFunction": p_dpf,
        "Age": p_age
    }

    # Run Prediction
    result = predictor.predict_single(patient_data, model_name=model_choice)
    prob = result["risk_probability"]
    percentage = result["risk_percentage"]
    tier = result["risk_tier"]

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 25px 0;'>", unsafe_allow_html=True)
    st.markdown("### 📊 Assessment Outcome & Diagnostic Risk Report")

    col_res1, col_res2 = st.columns([1.1, 1.3])

    with col_res1:
        # Gauge Meter for Diabetes Risk
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'size': 44, 'family': 'Plus Jakarta Sans', 'color': result["risk_color"]}},
            title={'text': f"<b>Diabetes Risk Probability</b><br><span style='font-size:0.85em;color:#94a3b8'>Classifier: {model_choice}</span>", 'font': {'size': 18, 'color': '#f8fafc'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': result["risk_color"], 'thickness': 0.28},
                'bgcolor': "rgba(30, 41, 59, 0.4)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.22)'},
                    {'range': [30, 65], 'color': 'rgba(245, 158, 11, 0.22)'},
                    {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.22)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.8,
                    'value': percentage
                }
            }
        ))
        gauge_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Risk Tier Banner
        if tier == "Low Risk":
            banner_class = "risk-banner-low"
            status_icon = "🟢"
            status_msg = "Patient is currently within low-risk metabolic parameters. Routine preventative health screening is advised."
        elif tier == "Moderate Risk":
            banner_class = "risk-banner-med"
            status_icon = "🟡"
            status_msg = "Elevated risk markers detected (Pre-Diabetes warning zone). Lifestyle interventions and lab monitoring are strongly indicated."
        else:
            banner_class = "risk-banner-high"
            status_icon = "🔴"
            status_msg = "High probability of Type 2 Diabetes or severe glucose dysregulation detected. Clinical evaluation and diagnostic blood work (HbA1c / OGTT) are recommended urgently."

        st.markdown(f"""
        <div class="{banner_class}">
            <div style="font-size: 1.3rem; font-weight: 700; color: {result['risk_color']}; margin-bottom: 6px;">
                {status_icon} Risk Category: {tier.upper()}
            </div>
            <div style="font-size: 0.95rem; color: #e2e8f0; line-height: 1.5;">
                {status_msg}
            </div>
            <div style="margin-top: 10px; font-size: 0.85rem; color: #94a3b8;">
                Predicted Binary Classification: <b>{"Diabetic (Class 1)" if result["prediction_class"] == 1 else "Non-Diabetic (Class 0)"}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_res2:
        st.markdown("#### 🧬 Patient Biomarkers vs. Healthy Baseline")
        st.caption("Percentage deviation of patient measurements compared to the median of healthy non-diabetic individuals.")

        # Deviation chart
        factor_df = result["factor_df"].sort_values(by="Percentage Deviation", ascending=True)
        colors = ['#ef4444' if x > 25 else ('#f59e0b' if x > 5 else '#10b981') for x in factor_df["Percentage Deviation"]]

        fig_dev = go.Figure(go.Bar(
            x=factor_df["Percentage Deviation"],
            y=factor_df["Feature"],
            orientation='h',
            marker_color=colors,
            text=[f"{val:+.1f}%" for val in factor_df["Percentage Deviation"]],
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>Deviation from Healthy: %{x:+.1f}%<extra></extra>"
        ))
        fig_dev.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=20, t=10, b=30),
            height=280,
            xaxis=dict(
                title="Deviation from Healthy Median (%)",
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.2)',
                gridcolor='rgba(255,255,255,0.05)',
                color='#94a3b8'
            ),
            yaxis=dict(color='#cbd5e1')
        )
        st.plotly_chart(fig_dev, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Actionable Clinical Recommendations
    st.markdown("### 💡 Personalized Clinical Insights & Action Plan")
    col_rec1, col_rec2 = st.columns(2)
    recs = result["recommendations"]

    for i, rec in enumerate(recs):
        target_col = col_rec1 if i % 2 == 0 else col_rec2
        with target_col:
            st.markdown(f"""
            <div class="rec-box">
                <div style="font-weight: 700; font-size: 1rem; color: #f8fafc; margin-bottom: 4px;">
                    {rec['icon']} {rec['title']}
                    <span style="float: right; font-size: 0.75rem; color: #38bdf8; text-transform: uppercase;">{rec['category']}</span>
                </div>
                <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.45;">
                    {rec['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Downloadable Clinical Summary Report
    st.markdown("<br>", unsafe_allow_html=True)
    report_text = f"""=====================================================
DIARISK AI - CLINICAL DIABETES RISK ASSESSMENT REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=====================================================

PATIENT BIOMARKERS:
- Age: {p_age} years
- Number of Pregnancies: {p_pregnancies}
- Fasting / 2-hr Plasma Glucose: {p_glucose} mg/dL
- Diastolic Blood Pressure: {p_bp} mm Hg
- Body Mass Index (BMI): {p_bmi} kg/m²
- 2-Hour Serum Insulin: {p_insulin} µU/mL
- Triceps Skinfold Thickness: {p_skin} mm
- Diabetes Pedigree Function: {p_dpf}

ASSESSMENT RESULTS:
- Supervised Learning Algorithm: {model_choice}
- Calculated Diabetes Risk: {percentage}%
- Diagnostic Risk Classification: {tier.upper()}
- Binary Class Prediction: {"Diabetic (1)" if result["prediction_class"] == 1 else "Non-Diabetic (0)"}

CLINICAL RECOMMENDATIONS:
"""
    for rec in recs:
        report_text += f"\n[{rec['category']}] {rec['title']}\n{rec['detail']}\n"

    report_text += """
=====================================================
DISCLAIMER: This diagnostic summary was synthesized by
a supervised machine learning algorithm for risk triage.
It should be verified with an official diagnostic lab test.
=====================================================
"""

    st.download_button(
        label="📥 Download Clinical Patient Report (.txt)",
        data=report_text,
        file_name=f"diabetes_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# ==========================================
# TAB 2: MODEL PERFORMANCE & BENCHMARKS
# ==========================================
with tabs[1]:
    st.markdown("### 🏆 Supervised Learning Model Benchmarks")
    st.markdown("""
    All models were trained on 80% stratified training data and evaluated on an independent 20% test set, 
    supplemented with **5-Fold Stratified Cross-Validation** to guarantee generalizability.
    """)

    metrics_data = predictor.metrics
    bench_rows = []
    for model_name, m in metrics_data.items():
        bench_rows.append({
            "Classifier": model_name,
            "Accuracy": m["Accuracy"],
            "5-Fold CV Accuracy": m["CV_Accuracy_Mean"],
            "Precision": m["Precision"],
            "Recall (Sensitivity)": m["Recall"],
            "F1-Score": m["F1_Score"],
            "ROC-AUC": m["ROC_AUC"]
        })

    bench_df = pd.DataFrame(bench_rows)

    # Format for display
    display_df = bench_df.copy()
    for col in ["Accuracy", "5-Fold CV Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score", "ROC-AUC"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # Interactive metric comparison bar chart
    st.markdown("#### 📊 Comparative Metric Analysis")
    metric_to_plot = st.selectbox(
        "Select Metric to Compare Across Models",
        options=["ROC-AUC", "Accuracy", "Recall (Sensitivity)", "Precision", "F1-Score"],
        index=0
    )

    col_map = {
        "ROC-AUC": "ROC_AUC",
        "Accuracy": "Accuracy",
        "Recall (Sensitivity)": "Recall",
        "Precision": "Precision",
        "F1-Score": "F1_Score"
    }

    plot_col = col_map[metric_to_plot]
    sorted_bench = bench_df.sort_values(by=plot_col, ascending=True)

    fig_bench = px.bar(
        sorted_bench,
        x=plot_col,
        y="Classifier",
        orientation='h',
        text=sorted_bench[plot_col].apply(lambda x: f"{x*100:.1f}%"),
        color=plot_col,
        color_continuous_scale="Tealgrn",
        labels={plot_col: metric_to_plot}
    )
    fig_bench.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320,
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis=dict(color='#94a3b8', tickformat=".0%", gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(color='#cbd5e1')
    )
    st.plotly_chart(fig_bench, use_container_width=True)

    col_bench_roc, col_bench_cm = st.columns([1.2, 1])

    with col_bench_roc:
        st.markdown("#### 📈 Multi-Model ROC Curves")
        roc_fig = go.Figure()

        # Add diagonal baseline
        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='#64748b', width=2),
            name='Random Chance (AUC = 0.50)'
        ))

        # Add curves
        for m_name, curve_data in predictor.roc_curves.items():
            auc_val = predictor.metrics[m_name]["ROC_AUC"]
            roc_fig.add_trace(go.Scatter(
                x=curve_data["fpr"],
                y=curve_data["tpr"],
                mode='lines',
                line=dict(width=2.5),
                name=f"{m_name} ({auc_val:.3f})"
            ))

        roc_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=380,
            margin=dict(l=10, r=10, t=20, b=20),
            xaxis=dict(title="False Positive Rate (1 - Specificity)", color='#94a3b8', gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title="True Positive Rate (Sensitivity / Recall)", color='#94a3b8', gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(font=dict(size=10, color='#e2e8f0'), bgcolor='rgba(15, 23, 42, 0.7)')
        )
        st.plotly_chart(roc_fig, use_container_width=True)

    with col_bench_cm:
        st.markdown("#### 🎯 Interactive Confusion Matrix")
        selected_cm_model = st.selectbox("Inspect Confusion Matrix for:", options=predictor.get_model_names(), index=0)
        cm = np.array(predictor.metrics[selected_cm_model]["Confusion_Matrix"])
        
        # Labels
        cm_text = [
            [f"True Neg (TN)<br><b>{cm[0,0]}</b>", f"False Pos (FP)<br><b>{cm[0,1]}</b>"],
            [f"False Neg (FN)<br><b>{cm[1,0]}</b>", f"True Pos (TP)<br><b>{cm[1,1]}</b>"]
        ]

        fig_cm = ff_fig = px.imshow(
            cm,
            text_auto=False,
            labels=dict(x="Predicted Condition", y="Actual Ground Truth", color="Cases"),
            x=['Non-Diabetic (0)', 'Diabetic (1)'],
            y=['Non-Diabetic (0)', 'Diabetic (1)'],
            color_continuous_scale="Blues"
        )
        fig_cm.update_traces(text=cm_text, texttemplate="%{text}")
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            xaxis=dict(color='#cbd5e1'),
            yaxis=dict(color='#cbd5e1')
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("""
    > [!TIP]
    > **Clinical AI Note:** In medical diagnosis, **Recall (Sensitivity)** is prioritized alongside ROC-AUC. 
    > A False Negative (failing to identify an individual with diabetes) carries severe clinical risk, 
    > whereas a False Positive prompts confirmatory lab blood testing with no permanent harm.
    """)

# ==========================================
# TAB 3: FEATURE IMPORTANCE & EXPLAINABILITY
# ==========================================
with tabs[2]:
    st.markdown("### 🧠 Model Explainability: What Drives Diabetes Risk?")
    st.markdown("""
    Machine learning models examine non-linear interactions across physiological markers. 
    Here we decompose which biomarkers contribute most heavily to risk stratification.
    """)

    feat_data = predictor.feature_importance
    features = feat_data["features"]
    rf_imp = feat_data["Random Forest"]
    gb_imp = feat_data["Gradient Boosting"]

    imp_df = pd.DataFrame({
        "Feature": features,
        "Random Forest Importance": rf_imp,
        "Gradient Boosting Importance": gb_imp
    }).sort_values(by="Random Forest Importance", ascending=True)

    col_imp1, col_imp2 = st.columns([1.3, 1])

    with col_imp1:
        st.markdown("#### 🌳 Ensemble Feature Importance (Gini / Impurity Reduction)")
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            y=imp_df["Feature"],
            x=imp_df["Random Forest Importance"],
            name="Random Forest",
            orientation='h',
            marker_color='#38bdf8'
        ))
        fig_imp.add_trace(go.Bar(
            y=imp_df["Feature"],
            x=imp_df["Gradient Boosting Importance"],
            name="Gradient Boosting",
            orientation='h',
            marker_color='#818cf8'
        ))
        fig_imp.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=380,
            margin=dict(l=10, r=10, t=20, b=20),
            xaxis=dict(title="Relative Importance Score", color='#94a3b8', gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(color='#cbd5e1'),
            legend=dict(font=dict(color='#e2e8f0'), bgcolor='rgba(15, 23, 42, 0.7)')
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_imp2:
        st.markdown("#### 🔬 Pathophysiological Explanations")
        st.markdown("""
        1. **Glucose (Plasma Concentration)**  
           *Dominant predictor across all models.* Elevated extracellular glucose indicates pancreatic beta-cell fatigue and impaired insulin-mediated cellular uptake.
        
        2. **BMI (Body Mass Index)**  
           Adipose tissue accumulation, particularly visceral adiposity, releases free fatty acids and inflammatory cytokines (TNF-alpha, IL-6) causing peripheral insulin resistance.
        
        3. **Age**  
           Age-related mitochondrial dysfunction and sarcopenia decrease muscle glucose sink capacity, escalating progressive risk.
        
        4. **Diabetes Pedigree Function (DPF)**  
           Captures polygenic inheritance and shared familial epigenetic/lifestyle tendencies.
        
        5. **Insulin & Blood Pressure**  
           Hyperinsulinemia often precedes overt hyperglycemia; elevated BP reflects concomitant metabolic syndrome microangiopathy.
        """)

# ==========================================
# TAB 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tabs[3]:
    st.markdown("### 🔍 Dataset Cohort & Exploratory Data Analysis")
    if raw_df is not None:
        col_eda1, col_eda2, col_eda3, col_eda4 = st.columns(4)
        with col_eda1:
            st.metric("Total Patients in Cohort", len(raw_df))
        with col_eda2:
            diabetic_count = int(raw_df['Outcome'].sum())
            st.metric("Diabetic Patients (Class 1)", f"{diabetic_count} ({diabetic_count/len(raw_df)*100:.1f}%)")
        with col_eda3:
            non_diabetic_count = len(raw_df) - diabetic_count
            st.metric("Non-Diabetic Patients (Class 0)", f"{non_diabetic_count} ({non_diabetic_count/len(raw_df)*100:.1f}%)")
        with col_eda4:
            st.metric("Biomarker Features", raw_df.shape[1] - 1)

        st.markdown("<br>", unsafe_allow_html=True)

        col_viz1, col_viz2 = st.columns(2)

        with col_viz1:
            st.markdown("#### 🔥 Biomarker Correlation Heatmap")
            corr = raw_df.corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_colorbar=dict(title="r")
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        with col_viz2:
            st.markdown("#### 🎯 Interactive Feature Scatter Explorer")
            feat_x = st.selectbox("X-Axis Feature", options=predictor.feature_names, index=1) # Glucose
            feat_y = st.selectbox("Y-Axis Feature", options=predictor.feature_names, index=5) # BMI

            scatter_df = raw_df.copy()
            scatter_df["Diagnosis"] = scatter_df["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"})

            fig_scatter = px.scatter(
                scatter_df,
                x=feat_x,
                y=feat_y,
                color="Diagnosis",
                color_discrete_map={"Non-Diabetic": "#10b981", "Diabetic": "#ef4444"},
                opacity=0.75,
                hover_data=["Age", "Glucose", "BMI"],
                trendline="ols"
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(color='#94a3b8', gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(color='#94a3b8', gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(font=dict(color='#e2e8f0'), bgcolor='rgba(15, 23, 42, 0.7)')
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Raw Dataset Table with Search
        st.markdown("#### 📄 Clinical Dataset Browser")
        show_outcome_filter = st.radio(
            "Filter by Clinical Outcome:",
            options=["All Patients", "Diabetic Patients Only (Outcome=1)", "Non-Diabetic Patients Only (Outcome=0)"],
            horizontal=True
        )

        filtered_table = raw_df.copy()
        if show_outcome_filter == "Diabetic Patients Only (Outcome=1)":
            filtered_table = filtered_table[filtered_table["Outcome"] == 1]
        elif show_outcome_filter == "Non-Diabetic Patients Only (Outcome=0)":
            filtered_table = filtered_table[filtered_table["Outcome"] == 0]

        st.dataframe(filtered_table, use_container_width=True, height=260)
    else:
        st.warning("Dataset file 'diabetes.csv' not found.")

# ==========================================
# TAB 5: BATCH PATIENT SCREENING
# ==========================================
with tabs[4]:
    st.markdown("### 📁 Batch Patient Risk Stratification")
    st.markdown("""
    Upload an institutional CSV file containing multiple patient records to perform automated batch risk scoring, 
    risk tier tagging, and export prioritized triage reports.
    """)

    # Download Template Button
    if os.path.exists("sample_batch.csv"):
        with open("sample_batch.csv", "rb") as f:
            template_bytes = f.read()
        st.download_button(
            label="📄 Download Sample Batch Patient CSV Template",
            data=template_bytes,
            file_name="sample_patient_records.csv",
            mime="text/csv",
            help="Download a pre-formatted template with 10 patient cases."
        )

    uploaded_file = st.file_uploader("Upload Patient Records (CSV format)", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_raw = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded {len(batch_raw)} patient records.")
            
            batch_model = st.selectbox(
                "Select Classifier for Batch Screening",
                options=predictor.get_model_names(),
                key="batch_model_selector"
            )

            # Run Predictions
            processed_batch = predictor.predict_batch(batch_raw, model_name=batch_model)

            # Summary Metrics for Batch
            total_cases = len(processed_batch)
            high_risk_cases = int((processed_batch["Risk_Category"] == "High Risk").sum())
            med_risk_cases = int((processed_batch["Risk_Category"] == "Moderate Risk").sum())
            low_risk_cases = int((processed_batch["Risk_Category"] == "Low Risk").sum())

            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                st.metric("Total Patients Screened", total_cases)
            with col_b2:
                st.metric("🔴 High Risk (Triage)", f"{high_risk_cases} ({high_risk_cases/total_cases*100:.1f}%)")
            with col_b3:
                st.metric("🟡 Moderate Risk (Review)", f"{med_risk_cases} ({med_risk_cases/total_cases*100:.1f}%)")
            with col_b4:
                st.metric("🟢 Low Risk", f"{low_risk_cases} ({low_risk_cases/total_cases*100:.1f}%)")

            # Display with styled rows
            st.markdown("#### 📊 Triage Results & Risk Scores")
            st.dataframe(processed_batch, use_container_width=True)

            # Export Button
            csv_export = processed_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Screened Patients with Risk Scores (CSV)",
                data=csv_export,
                file_name=f"screened_patients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 40px 0 20px 0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    <b>DiaRisk AI</b> — Supervised Learning for Medical Informatics & Risk Stratification • Developed with Python, Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)
