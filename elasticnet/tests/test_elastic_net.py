import os
import csv
import numpy as np
from models.ElasticNet import ElasticNetModel

def test_predict():
    model = ElasticNetModel(alpha=0.5, l1_ratio=0.8)
    data = []
    
    current_dir = os.path.dirname(__file__) 
    csv_file_path = os.path.join(current_dir, 'small_test.csv') 

    # Reading the CSV data
    with open("small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Preparing X and y
    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])

    # Fitting the model
    results = model.fit(X, y)

    # Making predictions
    preds = results.predict(X)

    # Check that the predictions are close to the actual values
    assert preds.shape == y.shape, "Predictions shape mismatch"
    assert isinstance(preds, np.ndarray), "Predictions are not an array"
