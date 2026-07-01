from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Create FastAPI app
app = FastAPI()

# Load trained model
model = joblib.load("models/student_score_predictor.pkl")


# Input data format
class StudentData(BaseModel):
    feature: list[float]


# Home API
@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is Running!"
    }


# Prediction API
@app.post("/predict")
def predict(data: StudentData):

    prediction = model.predict([data.feature])

    return {
        "Predicted Exam Score": float(prediction[0])
    }