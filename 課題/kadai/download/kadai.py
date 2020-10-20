import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def predict(read_csv):

    df_x = pd.read_csv('train_x.csv')
    df_y = pd.read_csv('train_y.csv')

    df_x1 = df_x.dropna(axis=1)
    df_x1 = df_x1.loc[:, ~(df_x1.nunique() == 1)]

    drop = ["お仕事No."]
    for column in df_x1.columns:
        if column not in read_csv.columns:
            drop.append(column)
        elif df_x1[column].dtype == object:
            le = LabelEncoder()
            le = le.fit(df_x1[column])
            df_x1[column] = le.transform(df_x1[column])

    df_x1 = df_x1.drop(drop, axis=1)

    X = df_x1
    y = df_y["応募数 合計"]

    X_array = np.array(X)
    y_array = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1)

    rfc = RandomForestRegressor(random_state=0)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    drop = ["お仕事No."]
    for column in read_csv.columns:
        if column not in X.columns:
            drop.append(column)
        elif read_csv[column].dtype == object:
            le = LabelEncoder()
            le = le.fit(read_csv[column])
            read_csv[column] = le.transform(read_csv[column])

    No = read_csv["お仕事No."]
    A1 = read_csv.drop(drop, axis=1)

    A_array = np.array(A1)
    pre_A_array = rfc.predict(A_array)

    submit = pd.DataFrame({"お仕事No.": No, "応募数 合計": pre_A_array})

    return submit
