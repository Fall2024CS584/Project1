import numpy as np


def fill_if_null(data):
    null_boy = np.array(data.columns[data.isnull().any()])
    for i in null_boy:
        data[i] = data[i].fillna(data[i].mean())
    return data


def check_null(data):
    if data.isnull().values.any():
        fill_if_null(data)
        print(data.isnull().sum())
    else:
        print(data.isnull().sum())


def XandY(data, dept):

    Y = data[dept].to_numpy()
    data.drop(dept, axis=1, inplace=True)
    X = data.to_numpy()

    return [X, Y]

