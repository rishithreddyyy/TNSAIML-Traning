# 🩺 DiaRisk AI — Diabetes Risk Prediction Using Supervised Learning

An end-to-end, clinical-grade Machine Learning web application for **Diabetes Risk Prediction** built with **Python**, **Scikit-Learn**, and **Streamlit**.

---

## 🌟 Key Features

1. **🩺 Patient Risk Assessment (Clinical Calculator)**:
   - Interactive biomarker inputs: Glucose, Insulin, BMI, Blood Pressure, Skinfold Thickness, Age, Pregnancies, and Diabetes Pedigree Function (DPF).
   - Instant presets: *🟢 Healthy Adult*, *🟡 Borderline / Pre-Diabetic*, *🔴 High-Risk Diabetic*, and *🎲 Random Patient Sample*.
   - Dynamic Risk Speedometer / Gauge Chart with triage color zones:
     - **Low Risk** (< 30%)
     - **Moderate Risk** (30% - 65%)
     - **High Risk** (> 65%)
   - **Biomarker Deviation Chart**: Compares the patient's vitals directly against healthy cohort median values.
   - **Personalized Clinical Action Plan**: Evidence-based nutrition, exercise, and diagnostic testing recommendations (HbA1c, OGTT, Lipid panel).
   - **Downloadable Clinical Report**: Generates a formatted text summary for medical records.

2. **📊 Supervised Learning Benchmarks & Model Comparison**:
   - Compares 6 supervised learning algorithms:
     - **Random Forest Classifier**
     - **Gradient Boosting Classifier**
     - **Logistic Regression**
     - **Support Vector Machine (Calibrated SVC)**
     - **K-Nearest Neighbors (KNN)**
     - **Voting Ensemble (Hybrid Soft-Voting Classifier)**
   - Metrics calculated on stratified test data: **Accuracy, 5-Fold Stratified Cross-Validation Accuracy, Precision, Recall (Sensitivity), F1-Score, and ROC-AUC**.
   - Interactive Multi-Model **ROC Curves** with individual AUC scores.
   - Interactive **Confusion Matrix** with True Positive / False Negative breakdown.

3. **🧠 Feature Importance & Model Explainability**:
   - Tree impurity reduction feature importances (Random Forest vs Gradient Boosting).
   - Pathophysiological rationale for why Glucose, BMI, and Age dominate diabetes risk.

4. **🔍 Exploratory Data Analysis (EDA)**:
   - Interactive correlation matrix heatmap.
   - Scatter plots with OLS regression trends (e.g. Glucose vs BMI vs Age).
   - Filterable raw clinical dataset viewer with outcome segmentation.

5. **📁 Batch Patient Screening**:
   - Institutional CSV upload for bulk risk scoring.
   - Downloadable CSV template with pre-formatted patient cases.
   - Batch summary statistics: Total screened, High-Risk triage count, Moderate, and Low Risk percentages.
   - One-click export of screened records with calculated risk scores and risk categories.

---

## 🚀 How to Run

### 1. Ensure Dependencies are Installed
```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn plotly joblib
```

### 2. Train Models (if updating dataset)
```bash
python train_models.py
```

### 3. Start the Web Application
```bash
streamlit run app.py --server.port 8502
```
Or simply double-click `run.bat` on Windows!

Open your browser at: **[http://localhost:8502](http://localhost:8502)**

---

## 📁 Project Structure

```
├── app.py                   # Streamlit interactive web application
├── predictor.py             # Preprocessing, model inference, and clinical guidance logic
├── train_models.py          # Training pipeline & metric evaluation for all supervised models
├── diabetes.csv             # Pima Indians Diabetes dataset
├── sample_batch.csv         # Sample batch patient template for bulk screening
├── run.bat                  # One-click Windows launch script
├── .streamlit/
│   └── config.toml          # Custom dark clinical UI theme configuration
└── models/                  # Saved models, scalers, imputers, and pre-computed metrics
    ├── trained_models.joblib
    ├── scaler.joblib
    ├── imputer.joblib
    ├── metrics.joblib
    ├── roc_curves.joblib
    ├── feature_importance.joblib
    ├── feature_names.joblib
    ├── reference_stats.joblib
    └── test_sample.csv
```

---

## ⚕️ Disclaimer
*DiaRisk AI is an algorithmic decision-support tool designed for screening, educational, and risk stratification purposes. Definitive medical diagnosis requires laboratory testing (Fasting Plasma Glucose, 2-Hour Oral Glucose Tolerance Test, or Glycated Hemoglobin HbA1c) evaluated by a licensed healthcare provider.*
