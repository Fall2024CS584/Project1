import csv
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from elasticnet.models.ElasticNet import ElasticNetModel

def load_data(filepath):
    data = []
    with open(filepath, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])
    return X, y

def test_predict():
    model = ElasticNetModel()

    train_X, train_y = load_data(os.path.join(project_root, 'train_data.csv'))

    test_X, test_y = load_data(os.path.join(project_root, 'test_data.csv'))

    model.fit(train_X, train_y)

    preds = model.predict(test_X)
    #print(f"prediction:\n {preds}")

    mse = np.mean((preds - test_y) ** 2)
    #print(mse)
    assert mse < 1

    print("Actual\tPredicted\tAbs Error")
    
    [print(f"{a}\t{p}\t{abs(a-p)}") for a, p in zip(test_y, preds)]

    print(f"MSE : {mse}")

if __name__ == "__main__":
    test_predict()
