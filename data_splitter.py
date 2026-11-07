from sklearn.model_selection import train_test_split

def split_data(X, y, task_type):
    
    if task_type == "classification":
        stratify=y
    elif task_type == "regression":
        stratify=None
    else:
        raise ValueError(f"ERROR! Unsupported task type: {task_type}!")
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )
    return X_train,X_test,y_train,y_test