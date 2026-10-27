
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Load the Portuguese dataset only
data = pd.read_csv("data/student-por.csv", sep=";")

features = ["G1", "G2", "absences", "failures"]
target = "G3"

X = data[features]
y = data[target]

# IMPORTANT:
# Use the SAME test_size and random_state as your training script.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 2. Load your existing saved model
saved_data = joblib.load(
    "models/student_grade_model_4features.joblib"
)
model = saved_data["model"]

# 3. Predict grades for the test students
predictions = model.predict(X_test)

# 4. Compare actual and predicted grades
results = X_test.copy()
results["Actual_G3"] = y_test
results["Predicted_G3"] = predictions
results["Absolute_Error"] = (
    results["Actual_G3"] - results["Predicted_G3"]
).abs()

# 5. Display the 10 largest prediction errors
worst_cases = results.sort_values(
    "Absolute_Error", ascending=False
).head(10)

print("\nTOP 10 LARGEST PREDICTION ERRORS")
print("=" * 75)
print(worst_cases.round(2).to_string())

print("\nOVERALL ERROR CHECK")
print("=" * 35)
print(
    "MAE:",
    round(mean_absolute_error(y_test, predictions), 3)
)
print("Test students:", len(y_test))

print("\nAnalysis complete. No model was retrained or overwritten.")
