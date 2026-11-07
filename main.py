from data_loader import load_dataset, split_features_target
from data_splitter import split_data
from data_cleaner import clean_values, clean_target, impute_features
from model import get_models, train
from evaluator import evaluate_models

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

path = input("enter a .CSV file path: ").strip()
separator = input("Enter the CSV separator (, or ;): ").strip()
target_column = input("Enter the target column's name: ").strip()
task_type = input('enter task type (classification / regression): ').strip().lower()

df = load_dataset(path, separator)

X, y = split_features_target(df, target_column)

X = clean_values(X)
X, y = clean_target(X, y)

X_train, X_test, y_train, y_test= split_data(X, y, task_type)

X_train, X_test, imputer = impute_features(X_train,X_test)

models = get_models(task_type)
models = train(models, X_train, y_train)
results = evaluate_models(models,X_test, y_test)
print(f"Succesfully trained {len(models)} models.")
print("Models:", list(models.keys()))

for model_name, metrics in results.items():
    print(f"\n {model_name}")
    for metric_name, score in metrics.items():
        print(f"{metric_name}: {score:.4f}")