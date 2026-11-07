from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        predictions = model.predict(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1_Score": f1_score(y_test, predictions, zero_division=0)
        }

    return results