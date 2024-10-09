import csv

import numpy as np
import pandas as pd
import pytest

from elasticnet.models.ElasticNet import ElasticNetModel


@pytest.fixture
def example_data():
    # Load the dataset using pandas
    df = pd.read_csv('small_test.csv')
    X = df.drop(columns=['y']).values  
    y = df['y'].values                
    
    return X, y

def test_fit(example_data):
    X, y = example_data

    # Split the data: 80% for training and 20% for testing
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Initialize model
    model = ElasticNetModel(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-4, epochs=1)

    # Fit the model on the training data
    results = model.fit(X_train, y_train)

    # Check if the coefficients and intercept are non-zero
    assert model.coef_ is not None, "Coefficients should not be None after fitting"
    assert model.intercept_ != 0.0, "Intercept should not be zero after fitting"

    # Test on the testing data
    predictions = results.predict(X_test)
    
    # Calculate R² score
    mse,r2=results.evaluate(X_test,y_test)
    print(f"R² score: {r2}")
    print(f"MSE: {mse}")
    
    # Assert that the R² score is greater than or equal to 0.80 and mse is less than 10
    assert r2 >= 0.80, f"Model performance is insufficient, R² score: {r2}"
    assert mse < 10, f"Model performance is insufficient, MSE: {mse}"

