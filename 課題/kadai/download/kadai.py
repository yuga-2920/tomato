import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def predict(test_x):

    df_x = pd.read_csv('train_x.csv')
    df_y = pd.read_csv('train_y.csv')
    test_x = pd.read_csv("test_x.csv")

    df = pd.merge(df_x, df_y, on="お仕事No.", how="outer")
    y = df["応募数 合計"]

    df = df.dropna(axis=1)
    df = df.loc[:, ~(df.nunique() == 1)]

    col = "給与/交通費　給与下限"
    mean = df[col].mean()
    sigma = df[col].mean()

    low = mean - 3 * sigma
    high = mean + 3 * sigma

    df = df[(df[col] > low) & (df[col] < high)]

    drop = ["お仕事No.", "応募数 合計"]
    for column in df.columns:
        if column not in test_x.columns:
            drop.append(column)
        elif df[column].dtype == object:
            le = LabelEncoder()
            le = le.fit(df[column])
            df[column] = le.transform(df[column])

    df = df.drop(drop, axis=1)

    X_array = np.array(df)
    y_array = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_array, test_size=0.4, random_state=1)

    rfc = RandomForestRegressor(random_state=0)
    rfc.fit(X_train, y_train)

    drop = ["お仕事No."]
    for column in test_x.columns:
        if column not in df.columns:
            drop.append(column)
        elif test_x[column].dtype == object:
            le = LabelEncoder()
            le = le.fit(test_x[column])
            test_x[column] = le.transform(test_x[column])

    No = test_x["お仕事No."]
    A = test_x.drop(drop, axis=1)

    A_array = np.array(A)
    A_pred = rfc.predict(A_array)
    
    submit = pd.DataFrame({"お仕事No.": No, "応募数 合計": A_pred})

    return submit
