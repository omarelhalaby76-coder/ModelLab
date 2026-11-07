from sklearn.impute import SimpleImputer
def clean_values(X):
    empty_columns = X.columns[X.isna().all()].tolist()
    if empty_columns:
        X = X.drop(columns=empty_columns)
        print(f"removed completely empty columns: {empty_columns}")
    return X
def clean_target(X, y):
    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]
    return X, y

def impute_features(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    return X_train, X_test, imputer