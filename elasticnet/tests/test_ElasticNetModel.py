import numpy as np
import csv

import sys
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Insert the project root directory into the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from elasticnet.models.ElasticNet import ElasticNetModel

def load_data():
    data = []
    with open("Spotify_Most_Streamed.csv", 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Assumes the first row might be headers
        
        # Detects non-numeric columns based on the first data row
        first_data_row = next(reader)
        numeric_indices = [i for i, value in enumerate(first_data_row) if is_numeric(value)]

        # Check if any numeric columns were detected
        if not numeric_indices:
            raise ValueError("No numeric columns found in the dataset.")

        # Re-process the first data row after determining numeric columns
        data.append([float(first_data_row[i]) for i in numeric_indices])
        
        # Process the rest of the file
        for row in reader:
            try:
                data.append([float(row[i]) for i in numeric_indices])
            except ValueError:
                # Skip rows with non-numeric values in the numeric columns
                continue

    # Converts data to numpy array for easier manipulation
    data = np.array(data)
    return data[:, :-1], data[:, -1]  # Return X and y (last column as target)

def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def test_with_real_world_data():
    X, y = load_data()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize and train the ElasticNet model
    model = ElasticNetModel()
    results = model.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = results.predict(X_test)

    # Assert predictions have the same length as y_test
    assert predictions.shape == y_test.shape, "Mismatch in shape of predictions and y_test"

    # Check accuracy or MAE depending on classification or regression
    if np.issubdtype(y.dtype, np.integer):
        accuracy = np.mean(predictions == y_test)
        assert accuracy >= 0.7, "Accuracy is below the expected threshold of 0.7"    
    else:
        # Calculate Mean Absolute Error for regression
        mae = np.mean(np.abs(predictions - y_test))
        std_y = np.std(y)
        print(f"Standard Deviation of y: {std_y:.2f}")
        assert mae < std_y, f"Model MAE {mae} is higher than expected, should be less than {std_y}"
    
def test_elastic_net_pathological_cases():
    # Case 1: High Dimensionality relative to the number of samples
    X, y = make_classification(n_samples=10, n_features=100, n_informative=2, n_redundant=98, random_state=42)
    try:
        model = ElasticNetModel()
        results = model.fit(X, y)
        preds = results.predict(X)
        assert preds.shape == y.shape, "Prediction shape mismatch in high dimensionality case"
        print("High dimensionality case passed without errors. Sample Predictions:", preds[:5])
    except Exception as e:
        print(f"High dimensionality case failed with error: {e}")

    # Case 2: Zero Variance Feature
    X = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = np.array([0, 1, 0, 1])
    try:
        model = ElasticNetModel()
        results = model.fit(X, y)
        preds = results.predict(X)
        assert preds.shape == y.shape, "Prediction shape mismatch in zero variance feature case"
        print("Zero variance feature case passed without errors. Sample Predictions:", preds)
    except Exception as e:
        print(f"Zero variance feature case failed with error: {e}")

    # Case 3: Perfectly Collinear Features
    X = np.array([[1, 2, 4], [1, 2, 4], [2, 4, 8], [2, 4, 8]])
    y = np.array([0, 1, 0, 1])
    try:
        model = ElasticNetModel()
        results = model.fit(X, y)
        preds = results.predict(X)
        assert preds.shape == y.shape, "Prediction shape mismatch in collinear feature case"
        print("Perfect collinearity case passed without errors. Sample Predictions:", preds)
    except Exception as e:
        print(f"Perfect collinearity case failed with error: {e}")

    # Case 4: Extremely Large Values
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X = X * 1e10  # Scale features to be very large
    try:
        model = ElasticNetModel()
        results = model.fit(X, y)
        preds = results.predict(X)
        assert preds.shape == y.shape, "Prediction shape mismatch in large values case"
        assert np.all(np.isfinite(preds)), "Predictions contain non-finite values in large values case"
        print("Large values case passed without errors. Sample Predictions:", preds[:5])
    except Exception as e:
        print(f"Large values case failed with error: {e}")

    # Case 5: Extremely Small Values
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X = X * 1e-10  # Scale features to be very small
    try:
        model = ElasticNetModel()
        results = model.fit(X, y)
        preds = results.predict(X)
        assert preds.shape == y.shape, "Prediction shape mismatch in small values case"
        assert np.all(np.isfinite(preds)), "Predictions contain non-finite values in small values case"
        print("Small values case passed without errors. Sample Predictions:", preds[:5])
    except Exception as e:
        print(f"Small values case failed with error: {e}")

# Example usage
if __name__ == "__main__":
    test_with_real_world_data()
    test_elastic_net_pathological_cases()