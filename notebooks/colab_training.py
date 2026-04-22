# !pip install scikit-learn pandas numpy joblib imbalanced-learn xgboost

import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from imblearn.over_sampling import SMOTE

# Optional: XGBoost (often outperforms RF on tabular data)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
    print("✅ XGBoost available")
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed. Using RandomForest + GradientBoosting only.")


# ──────────────────────────────────────────────
# 2. Load Dataset
# ──────────────────────────────────────────────
# Update this path for your Colab environment
DATA_PATH = "Credit_Score_Classification_Dataset.csv"  # or Google Drive path
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['Credit Score'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nSample:\n{df.head()}")


# ──────────────────────────────────────────────
# 3. Feature Engineering
# ──────────────────────────────────────────────
# ⚠️ CRITICAL: These steps MUST match src/preprocessing.py exactly!

# 3a. Income per Dependent
df["Income_per_Dependent"] = df["Income"] / (df["Number of Children"] + 1)

# 3b. Age-Income Ratio
df["Age_Income_Ratio"] = df["Income"] / df["Age"]

print("\n✅ Engineered features created:")
print(f"  Income_per_Dependent range: [{df['Income_per_Dependent'].min():.0f}, {df['Income_per_Dependent'].max():.0f}]")
print(f"  Age_Income_Ratio range: [{df['Age_Income_Ratio'].min():.0f}, {df['Age_Income_Ratio'].max():.0f}]")


# ──────────────────────────────────────────────
# 4. Categorical Encoding
# ──────────────────────────────────────────────
# ⚠️ CRITICAL: These mappings MUST match src/config.py exactly!

# Education — ordinal (preserves hierarchy)
EDUCATION_ORDER = {
    "High School Diploma": 1,
    "Associate's Degree": 2,
    "Bachelor's Degree": 3,
    "Master's Degree": 4,
    "Doctorate": 5,
}
df["Education_Level"] = df["Education"].map(EDUCATION_ORDER)
df = df.drop(columns=["Education"])

# Gender — binary
GENDER_MAP = {"Female": 0, "Male": 1}
df["Gender"] = df["Gender"].map(GENDER_MAP)

# Marital Status — binary
MARITAL_STATUS_MAP = {"Single": 0, "Married": 1}
df["Marital_Status"] = df["Marital Status"].map(MARITAL_STATUS_MAP)
df = df.drop(columns=["Marital Status"])

# Home Ownership — binary
HOME_OWNERSHIP_MAP = {"Rented": 0, "Owned": 1}
df["Home_Ownership"] = df["Home Ownership"].map(HOME_OWNERSHIP_MAP)
df = df.drop(columns=["Home Ownership"])

# Number of Children — rename
df = df.rename(columns={"Number of Children": "Number_of_Children"})

print("\n✅ Categorical encoding complete")


# ──────────────────────────────────────────────
# 5. Target Encoding
# ──────────────────────────────────────────────
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df["Credit Score"])
df = df.drop(columns=["Credit Score"])

print(f"\nTarget classes: {list(target_encoder.classes_)}")
print(f"Encoded mapping: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")


# ──────────────────────────────────────────────
# 6. Feature Ordering
# ──────────────────────────────────────────────
# ⚠️ CRITICAL: This order MUST match src/config.py FINAL_FEATURE_ORDER!

FINAL_FEATURE_ORDER = [
    "Age",
    "Gender",
    "Income",
    "Education_Level",
    "Marital_Status",
    "Number_of_Children",
    "Home_Ownership",
    "Income_per_Dependent",
    "Age_Income_Ratio",
]

X = df[FINAL_FEATURE_ORDER]
print(f"\n✅ Feature matrix shape: {X.shape}")
print(f"Feature order: {list(X.columns)}")


# ──────────────────────────────────────────────
# 7. Train/Test Split (Stratified)
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")


# ──────────────────────────────────────────────
# 8. Scaling (Numerical Features Only)
# ──────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "Age",
    "Income",
    "Number_of_Children",
    "Income_per_Dependent",
    "Age_Income_Ratio",
]

scaler = StandardScaler()
X_train[NUMERICAL_FEATURES] = scaler.fit_transform(X_train[NUMERICAL_FEATURES])
X_test[NUMERICAL_FEATURES] = scaler.transform(X_test[NUMERICAL_FEATURES])

print("\n✅ Scaling applied to numerical features")


# ──────────────────────────────────────────────
# 9. SMOTE — Class Balancing (Training Data Only)
# ──────────────────────────────────────────────
print(f"\nBefore SMOTE — Training class distribution: {np.bincount(y_train)}")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"After SMOTE  — Training class distribution: {np.bincount(y_train_res)}")


# ──────────────────────────────────────────────
# 10. Model Training & Comparison
# ──────────────────────────────────────────────

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    ),
}

if HAS_XGBOOST:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=target_encoder.classes_)}")

    if f1 > best_score:
        best_score = f1
        best_model = model
        best_name = name

print(f"\n{'=' * 60}")
print(f"🏆 BEST MODEL: {best_name} (F1: {best_score:.4f})")
print(f"{'=' * 60}")


# ──────────────────────────────────────────────
# 11. Hyperparameter Tuning (Best Model)
# ──────────────────────────────────────────────
print(f"\n🔧 Fine-tuning {best_name}...")

if best_name == "RandomForest":
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
    }
elif best_name == "GradientBoosting":
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
    }
elif best_name == "XGBoost":
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
    }
else:
    param_grid = {}

if param_grid:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        best_model, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_res, y_train_res)
    best_model = grid_search.best_estimator_
    print(f"\n✅ Best params: {grid_search.best_params_}")
    print(f"✅ Best CV F1: {grid_search.best_score_:.4f}")

# Final evaluation
y_pred_final = best_model.predict(X_test)
print(f"\n📊 Final Test Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_final, average='weighted'):.4f}")
print(f"\n{classification_report(y_test, y_pred_final, target_names=target_encoder.classes_)}")


# ──────────────────────────────────────────────
# 12. Export Artifacts
# ──────────────────────────────────────────────
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 12a. Save trained model
model_path = os.path.join(ARTIFACTS_DIR, "credit_model.pkl")
joblib.dump(best_model, model_path)
print(f"\n✅ Model saved to {model_path}")

# 12b. Save scaler
scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved to {scaler_path}")

# 12c. Save target encoder
encoder_path = os.path.join(ARTIFACTS_DIR, "target_encoder.pkl")
joblib.dump(target_encoder, encoder_path)
print(f"✅ Target encoder saved to {encoder_path}")

# 12d. Save feature config
feature_config = {
    "feature_order": FINAL_FEATURE_ORDER,
    "numerical_features": NUMERICAL_FEATURES,
    "n_features": len(FINAL_FEATURE_ORDER),
    "target_classes": list(target_encoder.classes_),
    "best_model_name": best_name,
    "best_f1_score": round(best_score, 4),
}
config_path = os.path.join(ARTIFACTS_DIR, "feature_config.json")
with open(config_path, "w") as f:
    json.dump(feature_config, f, indent=2)
print(f"✅ Feature config saved to {config_path}")

print(f"\n{'=' * 60}")
print("🎉 ALL ARTIFACTS EXPORTED SUCCESSFULLY!")
print(f"{'=' * 60}")
print(f"\n📁 Download these files from '{ARTIFACTS_DIR}/':")
print(f"   1. credit_model.pkl")
print(f"   2. scaler.pkl")
print(f"   3. target_encoder.pkl")
print(f"   4. feature_config.json")
print(f"\n📂 Place them in your local 'models/' folder and restart the API.")
