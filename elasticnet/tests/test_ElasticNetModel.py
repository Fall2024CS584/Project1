import os
import csv
import numpy as np
from elasticnet.models.ElasticNet import ElasticNetModel

def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Shuffle indices
    indices = np.random.permutation(len(y))
    test_set_size = int(len(y) * test_size)
    
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def test_predict():
    data = []
    csv_path = os.path.join(os.path.dirname(__file__), "small_test.csv")
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = np.array([[float(v) for k,v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])
    
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

    model = ElasticNetModel(alpha=0.5, l1_ratio=0.5, max_iter=10000)
    results = model.fit(X_train,y_train)
    preds = results.predict(X_test)
    print("Predicted values:", preds)
    print("Actual values:", y_test)
    print("Differences:", np.abs(preds - y_test))
    results.plot_loss_history()
    results.print_summary()

    tolerance = 15  
    assert np.all(np.abs(preds - y_test) < tolerance), "Predictions do not match expected values within the tolerance."


    
if __name__ == "__main__":
    test_predict()

   

