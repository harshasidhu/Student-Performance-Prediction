
import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    RandomizedSearchCV
)
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# 1. LOAD DATA
# ==================================================

data = pd.read_csv("data/student-por.csv", sep=";")

print("Dataset loaded:", data.shape)


# ==================================================
# 2. USE ONLY FOUR FEATURES
# ==================================================

feature_columns = ["G1", "G2", "absences", "failures"]

X = data[feature_columns]
y = data["G3"]


# ==================================================
# 3. CREATE TRAINING AND TEST SETS
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training students:", len(X_train))
print("Testing students:", len(X_test))


# ==================================================
# 4. SET UP CROSS-VALIDATION
# ==================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==================================================
# 5. DEFINE MODELS AND SETTINGS TO TEST
# ==================================================

search_spaces = {
    "Random Forest": (
        RandomForestRegressor(random_state=42, n_jobs=1),
        {
            "n_estimators": [100, 200, 400, 600],
            "max_depth": [None, 5, 8, 12, 16],
            "min_samples_split": [2, 4, 6, 10],
            "min_samples_leaf": [1, 2, 3, 4],
            "max_features": [1.0, 0.8, 0.6]
        }
    ),

    "Extra Trees": (
        ExtraTreesRegressor(random_state=42, n_jobs=1),
        {
            "n_estimators": [100, 200, 400, 600],
            "max_depth": [None, 5, 8, 12, 16],
            "min_samples_split": [2, 4, 6, 10],
            "min_samples_leaf": [1, 2, 3, 4],
            "max_features": [1.0, 0.8, 0.6]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.02, 0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.7, 0.85, 1.0]
        }
    )
}


# ==================================================
# 6. TUNE EACH MODEL USING TRAINING DATA ONLY
# ==================================================

best_models = {}
results = []

for name, (estimator, parameters) in search_spaces.items():

    print(f"\nTuning {name}...")

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameters,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y_train)

    best_models[name] = search.best_estimator_

    cv_mae = -search.best_score_

    print("Best cross-validation MAE:", round(cv_mae, 3))
    print("Best settings:", search.best_params_)

    results.append({
        "Model": name,
        "CV MAE": cv_mae
    })


# ==================================================
# 7. SELECT THE MODEL WITH THE LOWEST CV MAE
# ==================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("CV MAE")

print("\nCross-validation comparison")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["Model"]
best_model = best_models[best_name]

print("\nSelected model:", best_name)


# ==================================================
# 8. EVALUATE ON THE UNTOUCHED TEST SET
# ==================================================

predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nFinal Test Results")
print("------------------")
print("Selected model:", best_name)
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("R2:", round(r2, 3))


# ==================================================
# 9. COMPARE WITH A SIMPLE BASELINE
# ==================================================

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)
baseline_mae = mean_absolute_error(y_test, baseline_predictions)

print("Baseline MAE:", round(baseline_mae, 3))


# ==================================================
# 10. MEASURE CLOSE PREDICTIONS
# ==================================================

absolute_errors = abs(y_test.to_numpy() - predictions)

within_one = (absolute_errors <= 1).sum()
within_two = (absolute_errors <= 2).sum()
total = len(y_test)

print("\nPrediction Error Analysis")
print("-------------------------")
print(
    f"Within 1 grade point: {within_one}/{total} "
    f"({within_one / total * 100:.1f}%)"
)
print(
    f"Within 2 grade points: {within_two}/{total} "
    f"({within_two / total * 100:.1f}%)"
)
print("Largest absolute error:", round(absolute_errors.max(), 2))


# ==================================================
# 11. DISPLAY ACTUAL VS PREDICTED GRADES
# ==================================================

comparison = pd.DataFrame({
    "Actual Grade": y_test.to_numpy(),
    "Predicted Grade": predictions.round(2),
    "Absolute Error": absolute_errors.round(2)
})

print("\nFirst 15 test predictions:")
print(comparison.head(15).to_string(index=False))


# ==================================================
# 12. SAVE THE SELECTED MODEL
# ==================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "feature_columns": feature_columns,
        "model_name": best_name
    },
    "models/student_grade_model_4features.joblib"
)

print("\nBest model saved successfully!")
print("File: models/student_grade_model_4features.joblib")
