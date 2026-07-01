import joblib

model = joblib.load("../models/student_score_predictor.pkl")

print(model)