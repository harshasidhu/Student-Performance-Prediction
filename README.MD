# Student Performance Prediction

## 1. Project Overview

This project uses machine learning to predict a student's final Portuguese grade (`G3`) using four input features from the UCI Student Performance dataset.

The project includes a machine learning regression model and an interactive web application built with Streamlit. Users can enter student information and receive an estimated final grade on a scale of 0–20.

## 2. Objectives

- Predict students' final Portuguese grades using historical student data.
- Prepare and inspect the dataset.
- Compare machine learning regression algorithms.
- Train and evaluate a regression model.
- Analyze prediction errors.
- Develop an interactive web application using Streamlit.
- Demonstrate an end-to-end introductory machine learning workflow.

## 3. Technology Stack

The project uses the following technologies:

- **Python:** Main programming language.
- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical operations.
- **Scikit-learn:** Machine learning model training and evaluation.
- **Joblib:** Saving and loading the trained model.
- **Streamlit:** Building the interactive web application.

## 4. Dataset

The project uses the Portuguese-language student dataset from the UCI Student Performance dataset.

**Dataset file:** `data/student-por.csv`

Dataset details:

- Number of student records: 649
- Number of columns: 33
- Prediction target: `G3`
- Grade scale: 0–20

The mathematics dataset is not used to train the current Portuguese-grade prediction model.

The model uses four selected features to estimate the final Portuguese grade.

## 5. Input Features

The model uses the following four input features:

| Feature | Description |
|---|---|
| `G1` | First-period grade |
| `G2` | Second-period grade |
| `absences` | Number of student absences |
| `failures` | Number of previous class failures |

**Target variable:** `G3` — the student's final Portuguese grade.

The model predicts a numerical grade rather than a pass/fail classification.

## 6. Machine Learning Model

**Selected algorithm:** Extra Trees Regressor

Random Forest, Extra Trees, and Gradient Boosting were compared using cross-validation Mean Absolute Error (MAE).

The Extra Trees Regressor was selected based on the cross-validation results.

### Selected Model Parameters

| Parameter | Value |
|---|---:|
| `n_estimators` | 600 |
| `min_samples_split` | 4 |
| `min_samples_leaf` | 2 |
| `max_features` | 1.0 |
| `max_depth` | 5 |

### Saved Model

The trained model is stored at:

`models/student_grade_model_4features.joblib`

The saved model contains the trained regression estimator and the expected input feature names.

The Streamlit application loads this model to generate predictions without retraining it every time the application starts.

## 7. Model Evaluation

The selected model was evaluated on a held-out test set.

### Reported Evaluation Results

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 0.698 |
| Mean Squared Error (MSE) | 1.396 |
| R² Score | 0.857 |
| Predictions within 1 grade point | 83.1% |
| Predictions within 2 grade points | 97.7% |

### Interpretation

- **MAE:** The average absolute difference between the predicted and actual grades was approximately 0.698 grade points.
- **MSE:** The mean squared prediction error was 1.396.
- **R² Score:** The model explained approximately 85.7% of the variation in final grades on the reported test set.
- **Within 1 grade point:** Approximately 83.1% of predictions were within one point of the actual grade.
- **Within 2 grade points:** Approximately 97.7% of predictions were within two points of the actual grade.

These results describe the evaluated test split. They do not guarantee the same performance for future students or other populations.

## 8. Project Structure

The following structure reflects the main project files and the two saved regression model files.

```text
Student_Performance_Prediction/
├── app.py
├── train_model.py
├── predict_student.py
├── prepare_data.py
├── inspect_data.py
├── analyze_errors.py
├── investigate_case.py
├── requirements.txt
├── README.MD
├── data/
│   ├── student-por.csv
│   ├── student-mat.csv
│   ├── student-merge.R
│   └── student.txt
└── models/
    ├── student_grade_model.joblib
    └── student_grade_model_4features.joblib
```

### Main Files

- `app.py`: Runs the Streamlit web application.
- `train_model.py`: Trains and evaluates the machine learning model.
- `predict_student.py`: Generates a prediction for an individual student.
- `prepare_data.py`: Prepares the dataset for analysis and modelling.
- `inspect_data.py`: Inspects the dataset and its characteristics.
- `analyze_errors.py`: Analyzes prediction errors.
- `investigate_case.py`: Investigates a specific prediction case.
- `requirements.txt`: Lists the Python dependencies.
- `models/student_grade_model_4features.joblib`: Contains the saved four-feature regression model.

The original model file is retained separately. The current Streamlit application uses the four-feature model.

## 9. Installation and Setup

### Prerequisites

- Python installed on your computer.
- Visual Studio Code or another Python development environment.
- The project files and Portuguese dataset.

### Step 1: Open the Project Directory

Open a terminal in the project root directory:

```powershell
cd C:\Users\Sriha\Desktop\Student_Performance_Prediction
```

### Step 2: Create a Virtual Environment

If you do not already have a virtual environment, create one:

```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment

On Windows PowerShell, run:

```powershell
.\.venv\Scripts\Activate.ps1
```

If the virtual environment is already activated, you can skip this step.

### Step 4: Install Dependencies

Install the packages listed in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

### Step 5: Check Dependencies

Verify that the installed packages have no reported dependency conflicts:

```powershell
python -m pip check
```

## 10. Running the Web Application

Make sure your terminal is in the project root directory and your virtual environment is activated.

Run the following command:

```powershell
python -m streamlit run app.py
```

Streamlit will start the local web server and display a local URL in the terminal.

Open the displayed address in your browser. The default address is usually:

`http://localhost:8501`

### Using the Application

1. Open the Student Performance Prediction application.
2. Enter the student's first-period grade (`G1`).
3. Enter the student's second-period grade (`G2`).
4. Enter the number of absences.
5. Enter the number of previous class failures.
6. Submit the inputs to generate a prediction.
7. View the estimated final Portuguese grade on the 0–20 scale.

The application loads the saved model and displays the prediction without requiring the user to run the training script.

## 11. Limitations

- Predictions are estimates and are not guaranteed to match actual grades.
- The model uses only four input features.
- Students with unusual outcomes may receive inaccurate predictions.
- The dataset represents students from Portuguese schools, so performance may differ in other educational settings.
- The model may not accurately capture every factor that influences academic performance.
- The evaluation results apply to the reported test split and do not establish suitability for high-stakes educational decisions.
- The model should be used for educational demonstration and experimentation rather than as the sole basis for academic decisions.

## 12. Conclusion

The Student Performance Prediction project demonstrates an introductory end-to-end machine learning workflow, including dataset preparation, data inspection, model comparison, model evaluation, prediction-error analysis, model persistence, and web application deployment.

The final application uses an Extra Trees Regressor trained on the Portuguese student dataset to estimate final grades from four input features: `G1`, `G2`, `absences`, and `failures`.

The Streamlit interface makes the trained model accessible through a simple, interactive web application.