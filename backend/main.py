from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd


# --------------------------------------------------
# Create FastAPI app
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load trained ML model
# --------------------------------------------------

model = joblib.load("model.pkl")


# --------------------------------------------------
# Request data model
# --------------------------------------------------

class CustomerData(BaseModel):

    tenure: int = Field(
        ge=0,
        le=100,
        description="Customer tenure in months"
    )

    MonthlyCharges: float = Field(
        gt=0,
        le=1000,
        description="Monthly service charges"
    )

    TotalCharges: float = Field(
        ge=0,
        le=100000,
        description="Total amount charged to customer"
    )


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Customer Churn API is running"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_customer(data: CustomerData):

    # Convert request data into dictionary
    customer_dict = data.model_dump()

    # Convert dictionary into DataFrame
    customer_df = pd.DataFrame([customer_dict])

    # Make prediction
    prediction = model.predict(customer_df)

    # Get probability
    probability = model.predict_proba(customer_df)

    churn_probability = float(
        probability[0][1]
    )

    # Convert 0/1 prediction into Yes/No
    churn_prediction = (
        "Yes"
        if prediction[0] == 1
        else "No"
    )

    return {
        "churn_prediction": churn_prediction,
        "churn_probability": round(
            churn_probability,
            4
        )
    }