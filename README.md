# ModelLab

**A modular machine learning toolkit for training and comparing classification and regression models.**

ModelLab is a Python-based machine learning project designed to simplify the process of loading datasets, preparing data, training multiple machine learning models, and evaluating their performance.

It supports both classification and regression tasks through a modular architecture built with Scikit-learn.

## Features

- Load datasets from CSV files.
- Separate features and target variables.
- Validate datasets and detect common data issues.
- Handle missing values through data cleaning and imputation.
- Split datasets into training and testing sets.
- Train multiple machine learning models.
- Evaluate model performance using task-specific metrics.
- Compare different models using their evaluation results.

## Supported Models

### Classification

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### Regression

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Evaluation Metrics

### Classification

- Accuracy
- Precision
- Recall
- F1 Score

### Regression

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Installation

Install ModelLab using pip:

```bash
pip install model-lab-tool
```

Install dependencies manually if needed:

```bash
pip install pandas numpy scikit-learn
```

## Usage

ModelLab accepts a CSV dataset and allows you to specify:

1. The dataset file path.
2. The CSV separator.
3. The target column.
4. The machine learning task type:
   - `classification`
   - `regression`

The application then processes the dataset, trains the selected models, and evaluates their performance.

## Project Architecture

ModelLab follows a modular structure, separating different responsibilities into individual Python modules.

| Module | Responsibility |
|---|---|
| `main.py` | Main application workflow |
| `data_loader.py` | Dataset loading and feature-target separation |
| `data_cleaner.py` | Data cleaning and missing-value handling |
| `data_splitter.py` | Training and testing data splitting |
| `model.py` | Model selection and training |
| `evaluator.py` | Model evaluation |

## Dependencies

- Python
- Pandas
- NumPy
- Scikit-learn

## Project Status

ModelLab currently supports basic classification and regression workflows.

The project is under active development, and additional functionality may be introduced in future releases.

## License

See the LICENSE file for licensing information (MIT License).

---

**Developed as a modular machine learning project using Python and Scikit-learn.**