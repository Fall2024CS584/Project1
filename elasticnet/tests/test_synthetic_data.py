import numpy as np
import sys
import os
# Insert the project root directory into the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from elasticnet.models.ElasticNet import ElasticNetModel

def test_data():
    # Create synthetic data
    X = np.random.rand(100, 10)
    y = (np.sum(X, axis=1) > 5).astype(int)

    # Initialize and train the model
    model = ElasticNetModel(alpha=0.01, lambda1=0.05, lambda2=0.05, num_iterations=2000, batch_size=16)
    results=model.fit(X, y)

    # Make predictions
    X_new = np.random.rand(5, 10)
    predictions = results.predict(X_new)
    # Assert predictions have correct shape
    assert predictions.shape == (5,), "Unexpected shape of predictions"
    # Assert predictions are within binary range (0 or 1) for classification
    assert np.all((predictions == 0) | (predictions == 1)), "Predictions should be binary"


if __name__ == "__main__":
    test_data()