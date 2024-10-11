import numpy as np  
import pytest  # Importing pytest for testing
# tests/test_ElasticNetModel.py
import sys
import os

# Ensure that the models directory is in the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from ElasticNet import ElasticNet

@pytest.fixture
def setup_model():
    """Create and set up the model."""
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Instantiate the model with specified parameters
    return model  # Return the model to use in tests

def test_fit_and_predict(setup_model):
    """Verify if trained and make predictions."""
    model = setup_model  # Retrieve the model prepared by the setup fixture
    X = np.array([[1, 2], [2, 3], [3, 4]])  # Input data (features)
    y = np.array([1, 2, 3])  # Target values (what we want to predict)
    model.fit(X, y)  # Train the model using the feature data X and target variable y
    predictions = model.predict(X)  # Use the model to predict the same input
    
    expected = np.array([1, 2, 3])  # The expected predictions
    np.testing.assert_almost_equal(predictions, expected, decimal=1)  # Check if predictions match the expected values

def test_invalid_input_shape(setup_model):
    """Check whether error is raised when shapes do not match"""
    model = setup_model
    X = np.array([[1, 2], [2, 3]])  # Input data with two samples
    y = np.array([1])  # Invalid target, only one sample instead of two

    # Anticipate a ValueError when the number of samples in X does not align with those in y
    with pytest.raises(ValueError, match="Number of samples in X must match y."):  # Update the error message
        model.fit(X, y)  # Try to fit the model and expect it to fail


def test_no_variance(setup_model):
    """Check performance."""
    model = setup_model  # Use the instance directly, don't call it
    X = np.array([[1, 1], [1, 1], [1, 1]])  # Features with no variance
    y = np.array([1, 2, 3])  # Any target values

    # Check for an error when fitting the model with no variance:
    with pytest.raises(ValueError, match="Feature variance cannot be 0."):
        model.fit(X, y)  # Expecting a specific error when fitting

    # If you want to test predictions on valid input, you can do that separately:
    X_valid = np.array([[1, 2], [2, 3], [3, 4]])  # Features with variance
    y_valid = np.array([1, 2, 3]) 

    model.fit(X_valid, y_valid)  # Fit the model with valid data
    predictions = model.predict(X_valid)  # Make predictions

    assert predictions is not None  # Verify that predictions are generated and not null


def test_single_feature(setup_model):
    """Check capability"""
    model = setup_model  
    X = np.array([[1], [2], [3]])  # Input data with one feature
    y = np.array([1, 2, 3])  # Target values
    
    model.fit(X, y)  
    predictions = model.predict(X)  # Make predictions
    
    np.testing.assert_almost_equal(predictions, y, decimal=1)  # Check that predictions match the target

def test_fit_convergence(setup_model):
    """"Verify if the model reaches convergence when fitting a large dataset."""
    model = setup_model  
    X = np.random.rand(100, 2)  # Randomly generate 100 samples with 2 features
    y = X @ np.array([1.5, -2.0]) + 0.5 + np.random.normal(scale=0.1, size=100)  # Generate target values with some noise
    
    model.fit(X, y)  # Fit the model
    predictions = model.predict(X)  # Predict values
    
    # Evaluate model performance with some metrics
    mse, mae, r_squared = model.evaluate(X, y)  # Calculate metrics: mean squared error, mean absolute error, and R^2
    
    assert mse < 1.0  # Check that the mean squared error is reasonably low

def test_multiple_coefficients(setup_model):
    """Test the model with multiple coefficients."""
    model = setup_model  
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # Input data with two features
    y = np.array([2, 3, 4, 5]) 
    
    model.fit(X, y)  # Fit the model
    predictions = model.predict(X)  # Make predictions
    
    np.testing.assert_almost_equal(predictions, y, decimal=1)  # Check that predictions match the target

if __name__ == "__main__":
    pytest.main()  # Run all the tests when the script is executed
