# 🎓 Student Performance Analytics System

A Machine Learning project that predicts a student's exam score based on various academic, personal, and environmental factors. The project includes data preprocessing, model training, evaluation, model serialization, and deployment using FastAPI.

---

## 🚀 Project Overview,
a
This project uses **Linear Regression** to predict students' exam scores from features such as:

- Hours Studied
- Attendance
- Previous Scores
- Sleep Hours
- Internet Access
- Motivation Level
- Teacher Quality
- Family Income
- Physical Activity
- And more...

The trained model is deployed as a REST API using **FastAPI**, allowing users to send student information and receive predicted exam scores.

---

## 📂 Project Structure

```
Student-Performance-Analytics-System/
│
├── app/
│   └── main.py                  # FastAPI Application
│
├── data/
│   ├── StudentPerformanceFactors.csv
│   └── cleaned_students.csv
│
├── models/
│   └── student_score_predictor.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   └── train.py
|   └── predict.py
│   └── utils.py
|
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn

---

## 📊 Machine Learning Workflow

### 1. Data Loading
- Loaded the student performance dataset using Pandas.

### 2. Exploratory Data Analysis (EDA)
- Dataset overview
- Missing value analysis
- Statistical summary
- Distribution of Exam Scores

### 3. Data Preprocessing
- Filled missing values using mode.
- Encoded categorical variables using LabelEncoder.

### 4. Feature Engineering
Separated the dataset into:

- Features (X)
- Target Variable (y)

### 5. Train-Test Split

- Training Data: 80%
- Testing Data: 20%

### 6. Model Training

Model Used:

- Linear Regression

### 7. Model Evaluation

Metrics Used:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### 8. Model Saving

The trained model is saved using Joblib.

```
student_score_predictor.pkl
```

### 9. API Deployment

The model is deployed using FastAPI.

Available Endpoints:

```
GET /
```

Returns API status.

```
POST /predict
```

Predicts the student's exam score.

---

## 📈 Model Performance

| Metric | Value |
|---------|-------|
| MAE | 1.015
