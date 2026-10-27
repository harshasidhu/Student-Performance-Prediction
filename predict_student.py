
import joblib
import pandas as pd


# 1. Load the four-feature model
model_path = "models/student_grade_model_4features.joblib"

try:
    saved_data = joblib.load(model_path)
except FileNotFoundError:
    print("Model file not found!")
    print("First run: python train_model.py")
    raise SystemExit(1)

model = saved_data["model"]
feature_columns = saved_data["feature_columns"]

print("Four-feature model loaded successfully!")


# 2. Collect and validate the four inputs
def get_student_input():

    student = {}

    input_ranges = {
        "G1": (0, 20),
        "G2": (0, 20),
        "absences": (0, 100),
        "failures": (0, 4)
    }

    print("\nEnter the student's details.\n")

    for column in feature_columns:

        minimum, maximum = input_ranges[column]

        while True:

            value = input(
                f"{column} ({minimum}-{maximum}): "
            ).strip()

            try:
                number = float(value)

                if not number.is_integer():
                    print("Please enter a whole number.")
                    continue

                number = int(number)

                if minimum <= number <= maximum:
                    student[column] = number
                    break

                print(
                    f"Enter a value between {minimum} and {maximum}."
                )

            except ValueError:
                print("Invalid input. Please enter a number.")

    return pd.DataFrame([student], columns=feature_columns)


# 3. Predict the final grade
student_data = get_student_input()

predicted_grade = model.predict(student_data)[0]

predicted_grade = max(0, min(20, predicted_grade))


# 4. Display the result
print("\n===================================")
print("      STUDENT GRADE PREDICTION")
print("===================================")

print(f"G1 (First-period grade): {student_data.iloc[0]['G1']}")
print(f"G2 (Second-period grade): {student_data.iloc[0]['G2']}")
print(f"Absences: {student_data.iloc[0]['absences']}")
print(f"Previous failures: {student_data.iloc[0]['failures']}")

print(f"\nPredicted final grade (G3): {predicted_grade:.1f} / 20")

print("\nThis is an estimate, not a guaranteed grade.")
print("The model was trained on a Portuguese student dataset.")
print("Accuracy for other educational systems is unknown.")

print("===================================")
