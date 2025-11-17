import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def FunctionMain():
    try:
        df = pd.read_csv("churn.csv", sep=",")
    except Exception as e:
        print(f"Failed to read churn.csv: {e}")
        return

    print("DataFrame shape:", df.shape)
    print("Null counts:\n", df.isnull().sum())
    print("Exited value counts:\n", df["Exited"].value_counts())
    print("Unique CustomerId count:(to drop it later)", df["CustomerId"].nunique())
    print("Gender unique (raw):", df["Gender"].unique())

    # Normalize and encode Gender and map gender
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()
    print("Gender unique (normalized):", df["Gender"].unique())
    df["Gender"] = df["Gender"].map({"male": 0, "female": 1})

    '''One-hot encode Geography
    enc = preprocessing.OneHotEncoder(sparse=False)
    enc.fit(np.array(df["Geography"]).reshape(-1, 1))
    '''
    # Use pandas get_dummies for final dataframe
    df = pd.get_dummies(df, columns=["Geography"], dtype=int)

    # Drop unuseless columns
    df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True, errors="ignore")
    print("Columns after preprocessing:", df.columns.tolist())

    # Train/test split
    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.75, random_state=0, stratify=y
    )
    print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)

    # Decision Tree classifier
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("Decision Tree accuracy:", accuracy_score(y_test, y_pred))

    # New feature
    df["NewFeature"] = (df["EstimatedSalary"] * df["EstimatedSalary"] + df["Balance"]) / 2
    

if __name__ == "__main__":
    FunctionMain()