"""
Diabetes Risk Prediction - Model Training & Evaluation Pipeline
Supervised Learning using Multiple Classifiers
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

def build_and_train():
    print("Loading diabetes dataset...")
    df = pd.read_csv("diabetes.csv")
    print(f"Dataset shape: {df.shape}")

    # Columns where 0 indicates missing measurement
    zero_as_missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Copy dataset for preprocessing
    df_clean = df.copy()
    for col in zero_as_missing_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)

    # Impute missing values using median (robust against outliers)
    imputer = SimpleImputer(strategy='median')
    X_raw = df_clean.drop(columns=['Outcome'])
    y = df_clean['Outcome']
    feature_names = list(X_raw.columns)

    X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=feature_names)

    # Train / Test Split with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for feature name consistency
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    # Define Supervised Learning Models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Support Vector Machine": CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, random_state=42)),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9, weights='distance')
    }

    # Add soft-voting Ensemble Classifier combining top complementary models
    ensemble_clf = VotingClassifier(
        estimators=[
            ('rf', models["Random Forest"]),
            ('gb', models["Gradient Boosting"]),
            ('lr', models["Logistic Regression"]),
            ('svc', models["Support Vector Machine"])
        ],
        voting='soft'
    )
    models["Voting Ensemble (Hybrid)"] = ensemble_clf

    results = {}
    roc_curves = {}
    trained_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nTraining models and calculating evaluation metrics...")
    for name, model in models.items():
        # Fit on scaled training set
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

        # Test set predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # 5-Fold Cross Validation Accuracy
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        # ROC Curve points
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        results[name] = {
            "Accuracy": float(acc),
            "CV_Accuracy_Mean": float(cv_scores.mean()),
            "CV_Accuracy_Std": float(cv_scores.std()),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1_Score": float(f1),
            "ROC_AUC": float(auc),
            "Confusion_Matrix": cm.tolist()
        }

        roc_curves[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }

        print(f"[{name}] Acc: {acc:.3f} | Rec: {rec:.3f} | Prec: {prec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

    # Feature Importances from Tree models
    rf_importance = models["Random Forest"].feature_importances_.tolist()
    gb_importance = models["Gradient Boosting"].feature_importances_.tolist()
    lr_coefficients = models["Logistic Regression"].coef_[0].tolist()

    feature_analysis = {
        "features": feature_names,
        "Random Forest": rf_importance,
        "Gradient Boosting": gb_importance,
        "Logistic Regression Coef": lr_coefficients
    }

    # Reference statistics from clean data for patient comparison
    healthy_stats = df_clean[df_clean['Outcome'] == 0].describe().to_dict()
    diabetic_stats = df_clean[df_clean['Outcome'] == 1].describe().to_dict()
    overall_stats = df_clean.describe().to_dict()

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Save artifacts
    joblib.dump(imputer, "models/imputer.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(trained_models, "models/trained_models.joblib")
    joblib.dump(results, "models/metrics.joblib")
    joblib.dump(roc_curves, "models/roc_curves.joblib")
    joblib.dump(feature_analysis, "models/feature_importance.joblib")
    joblib.dump(feature_names, "models/feature_names.joblib")
    joblib.dump({
        "healthy": healthy_stats,
        "diabetic": diabetic_stats,
        "overall": overall_stats
    }, "models/reference_stats.joblib")

    # Also save test set for live interactive testing in app
    test_sample = X_test.copy()
    test_sample['Actual_Outcome'] = y_test
    test_sample.to_csv("models/test_sample.csv", index=False)

    print("\nAll models, metrics, and preprocessors saved to 'models/' successfully!")

if __name__ == "__main__":
    build_and_train()
