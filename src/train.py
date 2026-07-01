import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score

df = pd.read_csv("../data/StudentPerformanceFactors.csv")

#---------------- Handleing MIssing Values ------------------
df["Teacher_Quality"] = df["Teacher_Quality"].fillna(df["Teacher_Quality"].mode()[0])
df["Parental_Education_Level"]= df["Parental_Education_Level"].fillna(df["Parental_Education_Level"].mode()[0])
df["Distance_from_Home"] = df["Distance_from_Home"].fillna(df["Distance_from_Home"].mode()[0])

df.isnull().sum()

df.to_csv("../data/cleaned_students.csv", index=False)

Le = LabelEncoder()

categorical_columns = df.select_dtypes(include=["object", "string"]).columns
for columns in categorical_columns:
    df[columns] = Le.fit_transform(df[columns])
    print(df[columns])

x = df.drop("Exam_Score",axis=1)
y = df["Exam_Score"]

print(x.shape)
print(y.shape)

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size= 0.2 , random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = LinearRegression()

model.fit(x_train,y_train)
print(model)

y_predi = model.predict(x_test)
print(y_predi[:10])

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_predi
})

print(comparison.head(10))

mae = mean_absolute_error(y_test, y_predi)
mse = mean_squared_error(y_test, y_predi)
root = r2_score(y_test, y_predi)
rmse = np.sqrt(mse)

print("MAE :", mae)
print("MSE :", mse)
print("R2 Score:", root)
print("RMSE:", rmse)

joblib.dump(model,"../models/student_score_predictor.pkl")

print("Model saved successfully!")