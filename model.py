from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

def get_models(task_type):
    if task_type == "classification":
        models = {
            "Logistic_Regression": LogisticRegression(max_iter=10000),
            "Decision_Tree": DecisionTreeClassifier(),
            "Random_Forest": RandomForestClassifier()
        }
        return models
    elif task_type == "regression":
        models = {
            "Linear_Regression": LinearRegression(),
            "Decision_Tree": DecisionTreeRegressor(),
            "Random_Forest": RandomForestRegressor()
        }
        return models
    
    else:
        raise ValueError("Unsupported task type")
    
def train(models, X_train, y_train):
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models