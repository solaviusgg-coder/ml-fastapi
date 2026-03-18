import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import List

app = FastAPI(title="ML Prediction API")

# Модельді жасап сақтау
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
pipe.fit(X_train, y_train)
joblib.dump(pipe, "model.joblib")

model = joblib.load("model.joblib")

class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=4, max_length=4)

class PredictResponse(BaseModel):
    prediction: int
    model: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pred = model.predict([req.features])[0]
    return PredictResponse(prediction=int(pred), model="logreg+scaler")
