
import pandas as pd

# Load Portuguese dataset only
data = pd.read_csv("data/student-por.csv", sep=";")

# Inputs from the student with the largest prediction error
g1 = 11
g2 = 9
absences = 0
failures = 2

print("\n1. EXACT MATCHES")
print("=" * 50)

exact_matches = data[
    (data["G1"] == g1)
    & (data["G2"] == g2)
    & (data["absences"] == absences)
    & (data["failures"] == failures)
]

if exact_matches.empty:
    print("No students have exactly the same four inputs.")
else:
    print("Number of exact matches:", len(exact_matches))
    print("Final grades:")
    print(exact_matches["G3"].value_counts().sort_index())
    print("\nMatching records:")
    print(
        exact_matches[
            ["G1", "G2", "absences", "failures", "G3"]
        ].to_string(index=False)
    )

print("\n2. STUDENTS WITH FINAL GRADE ZERO")
print("=" * 50)

zero_grade = data[data["G3"] == 0]

print("Number of students with G3 = 0:", len(zero_grade))
print(
    zero_grade[
        ["G1", "G2", "absences", "failures", "G3"]
    ].to_string(index=False)
)

print("\n3. FINAL GRADE DISTRIBUTION")
print("=" * 50)
print(data["G3"].value_counts().sort_index())

print("\nInvestigation complete. No data or model was changed.")
