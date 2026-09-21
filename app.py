import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("data/customer_churn.csv")

print("Original dataset shape:", df.shape)


# =========================================================
# 2. CLEAN DATA
# =========================================================

# TotalCharges has some blank values
df = df[df["TotalCharges"] != " "].copy()

# Convert TotalCharges from string/object to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])


# =========================================================
# 3. CONVERT TARGET
# =========================================================

# Yes -> 1
# No  -> 0

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print("\nChurn values:")
print(df["Churn"].value_counts())

print("\nChurn datatype:")
print(df["Churn"].dtype)


# =========================================================
# 4. FEATURES (X) AND TARGET (y)
# =========================================================

X = df.drop("Churn", axis=1)

y = df["Churn"]


# customerID is just an identifier.
# It is not useful for predicting churn.

X = X.drop("customerID", axis=1)


print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget values:")
print(y.unique())


# =========================================================
# 5. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining shape:", X_train.shape)
print("Testing shape :", X_test.shape)


# =========================================================
# 6. CATEGORICAL COLUMNS
# =========================================================

categorical_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# =========================================================
# 7. NUMERICAL COLUMNS
# =========================================================

numerical_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


# =========================================================
# 8. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),

        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)


# =========================================================
# 9. COMPLETE ML PIPELINE
# =========================================================

model = Pipeline(
    steps=[
        
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            LogisticRegression(max_iter=10000)
        )
    ]
)


# =========================================================
# 10. TRAIN MODEL
# =========================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# =========================================================
# 11. PREDICTION
# =========================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# =========================================================
# 12. MODEL EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


print("\n===================================")
print("       MODEL PERFORMANCE")
print("===================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# =========================================================
# 13. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# =========================================================
# 14. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# =========================================================
# 15. SAVE COMPLETE PIPELINE
# =========================================================

joblib.dump(
    model,
    "model.pkl"
)

print("\n===================================")
print("Model saved successfully!")
print("File: model.pkl")
print("===================================")