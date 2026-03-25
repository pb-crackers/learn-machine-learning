"""
Lab 5.3 Starter — FastAPI Model Serving Application
=====================================================
This is the scaffolding for your prediction API. Customize the
PredictionRequest fields to match your dataset.

Run with:
    uvicorn starter:app --host 0.0.0.0 --port 8000 --reload

Then visit http://localhost:8000/docs for interactive API documentation.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# MODEL DEFINITION
# =============================================================================
# This must match the architecture used in Lab 5.2.
# In a real project, you would import this from a shared model.py file.

class TabularNet(nn.Module):
    """Feedforward neural network for tabular data."""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(checkpoint_path="models/final_model.pt"):
    """Load the trained PyTorch model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = TabularNet(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        output_dim=checkpoint["output_dim"],
        dropout_rate=checkpoint["dropout_rate"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Critical: disables dropout and uses running BN stats
    return model


def load_preprocessors():
    """Load the fitted scaler and imputer from Phase 1."""
    scaler = joblib.load("data/processed/scaler.joblib")
    imputer = joblib.load("data/processed/num_imputer.joblib")
    return scaler, imputer


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="ML Capstone Prediction API",
    description=(
        "Serves predictions from a trained model. "
        "Send a POST request to /predict with feature values as JSON."
    ),
    version="1.0.0",
)

# Global state — loaded at startup
model = None
scaler = None
imputer = None
feature_names = None


@app.on_event("startup")
def startup_event():
    """Load model and preprocessors when the server starts."""
    global model, scaler, imputer, feature_names

    model = load_model("models/final_model.pt")
    scaler, imputer = load_preprocessors()
    feature_names = list(pd.read_csv("data/processed/X_train.csv", nrows=0).columns)

    print(f"Model loaded successfully.")
    print(f"Expecting {len(feature_names)} features: {feature_names}")


# =============================================================================
# REQUEST / RESPONSE SCHEMAS
# =============================================================================
# IMPORTANT: Customize these fields to match YOUR dataset.
# The example below is for the Heart Disease dataset.
# Change field names, types, and constraints for your chosen dataset.

class PredictionRequest(BaseModel):
    """
    Input features for prediction.

    TODO: Replace these fields with your dataset's features.
    Add appropriate constraints (ge, le, etc.) for validation.
    """
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="0 = female, 1 = male")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression from exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Major vessels colored by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia type (0-3)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 55, "sex": 1, "cp": 2, "trestbps": 140, "chol": 250,
                "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
                "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2,
            }
        }


class PredictionResponse(BaseModel):
    """Response containing the model's prediction."""
    prediction: int = Field(..., description="Predicted class label (0 or 1)")
    probability: float = Field(..., description="Probability for the positive class")
    model_version: str = Field(default="1.0.0")


# =============================================================================
# PREDICTION LOGIC
# =============================================================================

def predict_from_raw(raw_input: dict) -> dict:
    """
    Preprocess raw input and run model inference.

    Parameters
    ----------
    raw_input : dict
        Feature name -> value mapping from the API request.

    Returns
    -------
    dict with "prediction" (int) and "probability" (float)
    """
    df = pd.DataFrame([raw_input])

    # Ensure correct column order and handle any missing columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
    df = df[feature_names]

    # Apply the same preprocessing as training
    df_imputed = pd.DataFrame(
        imputer.transform(df), columns=feature_names
    )
    df_scaled = pd.DataFrame(
        scaler.transform(df_imputed), columns=feature_names
    )

    # Run inference
    input_tensor = torch.tensor(df_scaled.values, dtype=torch.float32)

    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()
        prediction = int(probability > 0.5)

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
    }


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
def health_check():
    """
    Health check endpoint. Returns whether the server is running
    and the model is loaded.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest):
    """
    Make a prediction given input features.

    Send a JSON body with all required features. The response includes
    the predicted class and the probability. Invalid or out-of-range
    inputs will receive a 422 validation error.
    """
    try:
        raw_input = request.model_dump()
        result = predict_from_raw(raw_input)

        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# =============================================================================
# MAIN
# =============================================================================
# You can also run directly with: python starter.py
# But uvicorn is recommended for development.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("starter:app", host="0.0.0.0", port=8000, reload=True)
