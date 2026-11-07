import pandas as pd


def load_dataset(file_path, separator):
    df = pd.read_csv(file_path, sep=separator)

    if df.empty:
        raise ValueError("File is empty")

    if len(df) < 2:
        raise ValueError("File has less than 2 rows.")

    return df


def split_features_target(df, target_column):
    target_column = target_column.strip()

    if target_column not in df.columns:
        raise ValueError(
            f"Column '{target_column}' does not exist"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.isna().any().any():
        counts = X.isna().sum()
        print(f"Missing values detected in features:\n{counts[counts > 0]}")
    else:
        print("No missing values found in features.")

    if y.isna().any():
        print(f"Found {y.isna().sum()} missing values in the target column.")
    else:
        print("No missing values found in the target column.")

    non_num = X.select_dtypes(exclude="number")

    if non_num.empty:
        print("All features are numeric.")
    else:
        raise ValueError(
            f"Non-numeric values found in a column(s): {list(non_num.columns)}"
        )

    duplicates = df.duplicated().sum()

    if duplicates:
        raise ValueError(f"Duplicates found in the dataset: {duplicates}")
    else:
        print("No duplicates found.")

    return X, y