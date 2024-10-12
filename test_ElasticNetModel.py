import csv
import numpy as np
import matplotlib.pyplot as plt
from ElasticNet import ElasticNetModel

def mean_squared_error_manual(y_true, y_pred):
    # Ensure that y_true and y_pred are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute the squared differences
    squared_diffs = (y_true - y_pred) ** 2
    
    # Compute the mean of the squared differences
    mse = np.mean(squared_diffs)
    
    return mse

def test_predict():
    model = ElasticNetModel()
    data = []

    # Load data from CSV
    with open("/Users/singhdhruv/Documents/Development/github/Project1-ElasticNet/elasticnet/tests/BostonHousing.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Extract feature names (excluding the target variable 'medv')
    feature_names = [k for k in data[0].keys() if k != 'medv']

    # Convert data to float and construct X and y
    X = np.array([[float(datum[k]) for k in feature_names] for datum in data])
    y = np.array([float(datum['medv']) for datum in data])

    # Data checks
    assert not np.isnan(X).any(), "X contains NaN values."
    assert not np.isinf(X).any(), "X contains infinite values."
    assert not np.isnan(y).any(), "y contains NaN values."
    assert not np.isinf(y).any(), "y contains infinite values."
    variances = np.var(X, axis=0)
    assert not np.any(variances == 0), "One or more features have zero variance."

    # Manually standardize features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0, ddof=0)  # Population standard deviation

    # To prevent division by zero in case of zero variance
    X_std_nonzero = np.where(X_std == 0, 1, X_std)

    # Standardize the features
    X_scaled = (X - X_mean) / X_std_nonzero

    # Fit the model on the standardized features
    results = model.fit(X_scaled, y)

    # Get the coefficients and intercept from the fitted model
    coefficients_scaled = results.coef_
    intercept_scaled = results.intercept_

    # Adjust coefficients back to the original scale
    coefficients = coefficients_scaled / X_std_nonzero

    # Adjust the intercept
    intercept = intercept_scaled - np.dot(X_mean / X_std_nonzero, coefficients_scaled)

    # Make predictions using the original (unstandardized) features
    preds = np.dot(X, coefficients) + intercept

    # Assertions
    assert preds.shape == y.shape, "Predictions and true values have different shapes."
    assert np.all(np.isfinite(preds)), "Predictions contain non-finite values."

    # Compute Mean Squared Error
    mse = mean_squared_error_manual(y, preds)
    print(f"Mean Squared Error: {mse}")

    # Pair feature names with their coefficients
    feature_importance = dict(zip(feature_names, coefficients))

    # Display the feature importance
    print("Feature Importances:")
    for feature, coef in feature_importance.items():
        print(f"{feature}: {coef}")

    # Plotting
    plt.scatter(y, preds)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r')
    plt.show()
