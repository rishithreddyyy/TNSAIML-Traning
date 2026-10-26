"""
Predictor & Clinical Assessment Engine for Diabetes Risk Prediction
"""

import os
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

class DiabetesPredictor:
    def __init__(self):
        self.imputer = joblib.load(os.path.join(MODELS_DIR, "imputer.joblib"))
        self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        self.models = joblib.load(os.path.join(MODELS_DIR, "trained_models.joblib"))
        self.metrics = joblib.load(os.path.join(MODELS_DIR, "metrics.joblib"))
        self.roc_curves = joblib.load(os.path.join(MODELS_DIR, "roc_curves.joblib"))
        self.feature_importance = joblib.load(os.path.join(MODELS_DIR, "feature_importance.joblib"))
        self.feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
        self.reference_stats = joblib.load(os.path.join(MODELS_DIR, "reference_stats.joblib"))

    def get_model_names(self):
        return list(self.models.keys())

    def preprocess_input(self, input_df):
        """
        Preprocesses raw patient input: replaces zero values with NaN for applicable
        features, runs median imputer, and scales with fitted standard scaler.
        """
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df_clean = input_df.copy()
        for col in zero_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace(0, np.nan)

        # Impute
        imputed_data = self.imputer.transform(df_clean[self.feature_names])
        # Scale
        scaled_data = self.scaler.transform(imputed_data)
        return scaled_data, imputed_data

    def predict_single(self, patient_dict, model_name="Random Forest"):
        """
        Evaluates risk for a single patient dictionary.
        Returns risk score, risk level, clinical insights, and comparisons.
        """
        if model_name not in self.models:
            model_name = "Random Forest"

        model = self.models[model_name]
        df_input = pd.DataFrame([patient_dict])[self.feature_names]
        scaled_features, imputed_features = self.preprocess_input(df_input)

        # Prediction & Probability
        pred_class = int(model.predict(scaled_features)[0])
        prob_diabetic = float(model.predict_proba(scaled_features)[0][1])

        # Categorize Risk
        if prob_diabetic < 0.30:
            risk_tier = "Low Risk"
            risk_color = "#10B981" # Emerald Green
            risk_badge = "success"
        elif prob_diabetic < 0.65:
            risk_tier = "Moderate Risk"
            risk_color = "#F59E0B" # Amber / Orange
            risk_badge = "warning"
        else:
            risk_tier = "High Risk"
            risk_color = "#EF4444" # Rose / Crimson Red
            risk_badge = "danger"

        # Compare Patient Biomarkers against Healthy & Diabetic Baselines
        healthy_means = {k: self.reference_stats["healthy"][k]["50%"] for k in self.feature_names}
        patient_imputed_vals = {self.feature_names[i]: imputed_features[0][i] for i in range(len(self.feature_names))}

        factor_analysis = []
        for feat in self.feature_names:
            val = patient_imputed_vals[feat]
            healthy_median = healthy_means[feat]
            pct_diff = ((val - healthy_median) / (healthy_median + 1e-5)) * 100
            factor_analysis.append({
                "Feature": feat,
                "Patient Value": round(val, 2),
                "Healthy Reference (Median)": round(healthy_median, 2),
                "Percentage Deviation": round(pct_diff, 1)
            })

        factor_df = pd.DataFrame(factor_analysis)

        # Clinical Recommendations
        recommendations = self._generate_recommendations(patient_dict, prob_diabetic)

        return {
            "model_used": model_name,
            "prediction_class": pred_class,
            "risk_probability": prob_diabetic,
            "risk_percentage": round(prob_diabetic * 100, 1),
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "risk_badge": risk_badge,
            "factor_df": factor_df,
            "recommendations": recommendations,
            "imputed_values": patient_imputed_vals
        }

    def predict_batch(self, batch_df, model_name="Random Forest"):
        """
        Evaluates risk for a batch DataFrame of patients.
        """
        if model_name not in self.models:
            model_name = "Random Forest"
        model = self.models[model_name]

        # Verify columns exist
        missing_cols = [c for c in self.feature_names if c not in batch_df.columns]
        if missing_cols:
            raise ValueError(f"Batch dataset missing required columns: {missing_cols}")

        scaled_features, _ = self.preprocess_input(batch_df[self.feature_names])
        preds = model.predict(scaled_features)
        probs = model.predict_proba(scaled_features)[:, 1]

        result_df = batch_df.copy()
        result_df["Risk_Score"] = np.round(probs * 100, 1)
        result_df["Predicted_Class"] = preds
        result_df["Risk_Category"] = [
            "Low Risk" if p < 0.30 else ("Moderate Risk" if p < 0.65 else "High Risk")
            for p in probs
        ]
        return result_df

    def _generate_recommendations(self, patient, risk_prob):
        recs = []
        glucose = patient.get("Glucose", 0)
        bmi = patient.get("BMI", 0)
        bp = patient.get("BloodPressure", 0)
        age = patient.get("Age", 0)
        dpf = patient.get("DiabetesPedigreeFunction", 0)

        # Glucose specific
        if glucose >= 140:
            recs.append({
                "category": "Immediate Medical Attention",
                "icon": "⚠️",
                "title": "Elevated Blood Glucose Level",
                "detail": f"Patient glucose is {glucose} mg/dL (Pre-diabetic/Diabetic range >= 140 mg/dL). Immediate consultation with an endocrinologist for HbA1c testing is strongly advised."
            })
        elif glucose >= 100:
            recs.append({
                "category": "Metabolic Screening",
                "icon": "🩺",
                "title": "Impaired Fasting Glucose (Pre-Diabetes Range)",
                "detail": f"Patient glucose is {glucose} mg/dL (Normal range is 70-99 mg/dL). Monitor fasting blood sugar every 3 months."
            })

        # BMI specific
        if bmi >= 30:
            recs.append({
                "category": "Lifestyle & Weight Management",
                "icon": "🥗",
                "title": "Adiposity Reduction & Dietary Plan",
                "detail": f"Current BMI is {bmi:.1f} kg/m² (Obese tier >= 30). Clinically proven: 5-7% weight reduction reduces type 2 diabetes incidence by over 58%."
            })
        elif bmi >= 25:
            recs.append({
                "category": "Lifestyle & Nutrition",
                "icon": "🏃",
                "title": "Weight Optimization",
                "detail": f"Current BMI is {bmi:.1f} kg/m² (Overweight tier 25-29.9). Target 150 minutes of moderate-intensity aerobic activity per week."
            })

        # Blood pressure
        if bp >= 85:
            recs.append({
                "category": "Cardiovascular Health",
                "icon": "❤️",
                "title": "Elevated Diastolic Blood Pressure",
                "detail": f"Blood pressure is {bp} mm Hg. Hypertension exacerbates microvascular diabetes complications. Restrict dietary sodium (<2,300 mg/day)."
            })

        # Genetics
        if dpf > 0.5:
            recs.append({
                "category": "Genetic Predisposition",
                "icon": "🧬",
                "title": "Strong Family Genetic History",
                "detail": f"Diabetes Pedigree Function is {dpf:.3f} (above average). Heightened vigilance and annual screening are essential."
            })

        # General cadence based on overall risk
        if risk_prob >= 0.65:
            recs.append({
                "category": "Clinical Follow-up",
                "icon": "📋",
                "title": "Comprehensive Diagnostic Workup",
                "detail": "Perform Oral Glucose Tolerance Test (OGTT), Glycated Hemoglobin (HbA1c), Fasting Lipid Profile, and Renal Function Panel within 2-4 weeks."
            })
        elif risk_prob >= 0.30:
            recs.append({
                "category": "Preventative Care",
                "icon": "🛡️",
                "title": "Lifestyle Modification & Bi-Annual Review",
                "detail": "Adopt Mediterranean or DASH dietary pattern rich in whole grains and lean proteins. Retest metabolic panel in 6 months."
            })
        else:
            recs.append({
                "category": "Wellness Maintenance",
                "icon": "✅",
                "title": "Healthy Maintenance Profile",
                "detail": "Biomarkers are within healthy ranges. Continue routine annual physical checkups, balanced nutrition, and active lifestyle."
            })

        return recs
