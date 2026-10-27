import pandas as pd

data = pd.read_csv("data/student-por.csv", sep=";")

print("Dataset loaded successfully!")
print("\nNumber of rows and columns:", data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst five rows:")
print(data.head())

print("\nMissing values in each column:")
print(data.isnull().sum())
print("\nNumber of duplicate rows:", data.duplicated().sum())

print("\nDataset information:")
data.info()

print("\nNumerical summary:")
print(data.describe())

print(data.describe(include="object"))


import matplotlib.pyplot as plt

# 1. Distribution of final grades
plt.figure(figsize=(8, 5))
plt.hist(data["G3"], bins=range(0, 21), edgecolor="black")
plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# 2. Average final grade by study time
average_grades = data.groupby("studytime")["G3"].mean()

plt.figure(figsize=(8, 5))
average_grades.plot(kind="bar", edgecolor="black")
plt.title("Average Final Grade by Study Time")
plt.xlabel("Study Time Category (1–4)")
plt.ylabel("Average Final Grade")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. Absences compared with final grades
plt.figure(figsize=(8, 5))
plt.scatter(data["absences"], data["G3"], alpha=0.6)
plt.title("School Absences vs Final Grades")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.show()
