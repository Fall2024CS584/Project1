import csv

import numpy as np

from elasticnet.models.ElasticNet import MyElasticNetImplementation

def test_predict():
    model = MyElasticNetImplementation(learning_rate=0.01, iterations=1000, l1_penalty=0.01, l2_penalty=4)
    data = []
    with open("small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])  # Flatten y directly
    model.fit(X,y)
    preds = model.predict(X)
    print(preds)
    model.calculate_metrics(y, preds)

   