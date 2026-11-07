from .data_loader import load_dataset, split_features_target
from .data_splitter import split_data
from .data_cleaner import clean_values, clean_target, impute_features
from .model import get_models, train
from .evaluator import evaluate_models

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
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
    try:
        results = evaluate_models(models, X_test, y_test)
    except ValueError as error:
        # The evaluator's binary averaging is incompatible with multiclass
        # targets, such as the three classes in the Iris dataset.
        if task_type != "classification" or "average='binary'" not in str(error):
            raise

        results = {}
        for model_name, model in models.items():
            predictions = model.predict(X_test)
            results[model_name] = {
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
            }
    print(f"Succesfully trained {len(models)} models.")
    print("Models:", list(models.keys()))

    for model_name, metrics in results.items():
        print(f"\n {model_name}")
        for metric_name, score in metrics.items():
            print(f"{metric_name}: {score:.4f}")
            
if __name__ == "__main__":
    main()