
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
data = pd.read_csv("data/student-por.csv", sep=";")

# Step 2: Separate the target from the input features
X = data.drop(columns=["G3", "G1", "G2"])
y = data["G3"]

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 4: Check the results
print("Original dataset:", data.shape)
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Step 5: Identify numerical and categorical columns
numeric_features = X_train.select_dtypes(include=["number"]).columns
categorical_features = X_train.select_dtypes(exclude=["number"]).columns

# Step 6: Prepare a method to convert text into numbers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Step 7: Transform the training and testing data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Step 8: Check the results
print("Processed training features:", X_train_processed.shape)
print("Processed testing features:", X_test_processed.shape)
