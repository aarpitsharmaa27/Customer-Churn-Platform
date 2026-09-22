import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("data/customer_churn.csv")

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 2. Clean TotalCharges
# --------------------------------------------------

# Some rows contain a blank space in TotalCharges
df = df[df["TotalCharges"] != " "].copy()

# Convert TotalCharges from text to number
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])


# --------------------------------------------------
# 3. Convert Churn into 0/1
# --------------------------------------------------

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


print("\nChurn distribution:")
print(df["Churn"].value_counts())


# --------------------------------------------------
# 4. Select only generic customer features
# --------------------------------------------------

features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

X = df[features]

y = df["Churn"]


print("\nFeatures used by model:")
print(features)


# --------------------------------------------------
# 5. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 6. Create ML Pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(max_iter=10000)
        )
    ]
)


# --------------------------------------------------
# 7. Train Model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 8. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 9. Evaluate Model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

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


print("\n-----------------------------")
print("Model Performance")
print("-----------------------------")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 10. Save Model
# --------------------------------------------------

joblib.dump(model, "model.pkl")

print("\nModel saved successfully!")
print("Saved file: model.pkl")