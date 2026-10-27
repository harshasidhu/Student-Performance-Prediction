# Student Performance Prediction

An end-to-end machine learning project that predicts students' final Portuguese grades using academic performance data. The project includes data exploration, model training and evaluation, prediction-error analysis, and an interactive Streamlit application.

## Project Overview

This project uses machine learning to estimate a student's final Portuguese grade (`G3`) on a scale of 0–20.

The model uses four input features:
- First-period grade (`G1`)
- Second-period grade (`G2`)
- Number of absences (`absences`)
- Number of previous class failures (`failures`)

The application loads a previously trained model and generates predictions through a simple web interface.

## Objectives

- Explore and understand the student performance dataset.
- Inspect data quality and examine relationships between variables.
- Select relevant features for grade prediction.
- Compare machine learning regression algorithms.
- Train and evaluate a regression model.
- Analyze prediction errors and investigate individual cases.
- Save the trained model for reuse.
- Build an interactive prediction application using Streamlit.

## Technology Stack

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Scikit-learn** — Machine learning and model evaluation
- **Joblib** — Model persistence
- **Streamlit** — Interactive web application

## Dataset

The project uses the Portuguese-language dataset from the UCI Student Performance dataset.

**Dataset file:** `data/student-por.csv`

Dataset characteristics:
- Student records: 649
- Columns: 33
- Target variable: `G3`
- Grade scale: 0–20

The final prediction model uses the Portuguese dataset. The mathematics dataset is not required by the current application.

## Input Features

| Feature | Description |
|---|---|
| `G1` | First-period grade |
| `G2` | Second-period grade |
| `absences` | Number of school absences |
| `failures` | Number of previous class failures |
| `G3` | Final Portuguese grade to predict |

The model predicts a numerical grade, not a pass/fail classification.

## Machine Learning Model

**Selected algorithm:** Extra Trees Regressor

Random Forest, Extra Trees, and Gradient Boosting were compared using cross-validation Mean Absolute Error (MAE). The Extra Trees Regressor was selected based on the reported cross-validation results.

### Model parameters

| Parameter | Value |
|---|---:|
| `n_estimators` | 600 |
| `min_samples_split` | 4 |
| `min_samples_leaf` | 2 |
| `max_features` | 1.0 |
| `max_depth` | 5 |

**Saved model:** `models/student_grade_model_4features.joblib`

The Streamlit application loads this saved model rather than retraining it each time the application starts.

## Model Evaluation

The model was evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 0.698 |
| Mean Squared Error (MSE) | 1.396 |
| R² Score | 0.857 |
| Predictions within 1 grade point | 83.1% |
| Predictions within 2 grade points | 97.7% |

### Interpretation

- **MAE:** Predictions differed from actual grades by approximately 0.698 grade points on average.
- **MSE:** The mean squared prediction error was 1.396.
- **R²:** The model explained approximately 85.7% of the variation in final grades on the evaluated test set.
- **Within 1 point:** Approximately 83.1% of predictions were within one grade point of the actual grade.
- **Within 2 points:** Approximately 97.7% of predictions were within two grade points of the actual grade.

These results describe the evaluated test split. They do not guarantee the same performance on future students or in other educational settings.

## Project Structure

```text
Student-Performance-Prediction/
├── data/
│   ├── student-por.csv
│   └── student.txt
├── models/
│   └── student_grade_model_4features.joblib
├── .gitignore
├── README.MD
├── analyze_errors.py
├── app.py
├── inspect_data.py
├── investigate_case.py
├── predict_student.py
├── prepare_data.py
├── requirements.txt
└── train_model.py
```

### Main Files

| File | Purpose |
|---|---|
| `app.py` | Runs the Streamlit prediction application |
| `train_model.py` | Trains, compares, and evaluates regression models |
| `predict_student.py` | Generates an individual grade prediction |
| `prepare_data.py` | Prepares data for analysis and modelling |
| `inspect_data.py` | Inspects the dataset and creates exploratory visualizations |
| `analyze_errors.py` | Analyzes prediction errors |
| `investigate_case.py` | Investigates a specific prediction case |
| `data/student-por.csv` | Portuguese student performance dataset |
| `data/student.txt` | Dataset documentation |
| `models/student_grade_model_4features.joblib` | Saved four-feature regression model |
| `requirements.txt` | Python package dependencies |

## Installation and Setup

### Prerequisites

- Python installed on your computer
- Git
- Visual Studio Code or another Python development environment

### 1. Clone the repository

```bash
git clone https://github.com/harshasidhu/Student-Performance-Prediction.git
cd Student-Performance-Prediction
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If your virtual environment is already active, skip this step.

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Check dependencies

```powershell
python -m pip check
```

## Run the Streamlit Application

From the project root directory, run:

```powershell
python -m streamlit run app.py
```

Open the local URL shown in the terminal. By default, it is usually:

`http://localhost:8501`

### How to use the application

1. Enter the student's first-period grade (`G1`).
2. Enter the second-period grade (`G2`).
3. Enter the number of absences.
4. Enter the number of previous class failures.
5. Submit the inputs.
6. Review the estimated final Portuguese grade on the 0–20 scale.

The application uses the saved model to produce a prediction without requiring you to run the training script first.

## Limitations

- Predictions are estimates and may differ from actual grades.
- The model uses only four input features.
- Unusual student outcomes may be difficult to predict accurately.
- The dataset represents students from Portuguese schools, so results may not generalize to other educational settings.
- The reported evaluation metrics apply to the specific test split used in the project.
- The model has not been established as suitable for high-stakes educational decisions.

This project is intended for educational demonstration and experimentation, not as the sole basis for academic decisions.

## Future Improvements

- Add additional visualizations and an exploratory data analysis report.
- Compare additional regression algorithms and evaluation strategies.
- Improve the application's input validation and user experience.
- Deploy the Streamlit application for public access.
- Add automated tests for the prediction pipeline.
- Document future experiments and model improvements.

## Conclusion

This project demonstrates an introductory end-to-end machine learning workflow, including data inspection, exploratory analysis, feature selection, model comparison, evaluation, error analysis, model persistence, and interactive predictions.

The final application uses an Extra Trees Regressor to estimate students' final Portuguese grades from `G1`, `G2`, `absences`, and `failures`.