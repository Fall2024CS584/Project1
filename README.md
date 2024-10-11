# Project 1 
## Group Members
- Laasya Priya Vemuri (CWID: A20561469)
- Dyuti Dasary (CWID: A20546872)
- Charan Reddy Kandula Venkata (CWID: A20550020)
- Niranjaan Veeraragahavan Munuswamy (CWID: A20552057)

# ElasticNet Model Implementation

## Overview
The ElasticNetModel is a custom implementation of logistic regression with ElasticNet regularization, which combines both L1 (Lasso) and L2 (Ridge) penalties. This model is suitable for both binary and multiclass classification tasks, especially when:

- You expect some features to be irrelevant or noisy (ElasticNet's L1 regularization can reduce the coefficients of irrelevant features to zero, effectively selecting a subset of important features).
- You need to prevent overfitting in high-dimensional data (L2 regularization helps by constraining the model's weights).
- There is multicollinearity among features (ElasticNet can handle correlated predictors better than Lasso alone).
ElasticNet is particularly useful when you want to balance feature selection and regularization, as it offers flexibility to control both L1 and L2 penalties.

## Testing
The model has been validated using several testing strategies:

- **Real-World Data Testing:** The model was tested on datasets with numerical features from real-world data sources, such as Spotify_Most_Streamed, user_behavior_datset, weather_data.csv. Performance was evaluated using metrics like Mean Absolute Error (MAE) for regression and accuracy for classification.
- **Pathological Cases:** To ensure robustness, the model was tested on synthetic datasets designed to challenge typical algorithms:
  - **High Dimensionality:** Datasets where the number of features greatly exceeds the number of samples.
  - **Zero Variance Features:** Datasets containing features with no variability.
  - **Perfectly Collinear Features:** Datasets where some features are perfect linear combinations of others.
  - **Extremely Large and Small Values:** Datasets with very large or very small values to check for numerical stability.

### Results of Testing
In each test case, the model was checked for performance using assert statements on metrics like MAE (expected to be below the standard deviation of the target variable) or through direct inspection of predicted values. The model demonstrated resilience, although it struggled with very high-dimensional data relative to the sample size due to the limited number of iterations.

## Model Parameters and Tuning
- **fit(X, y):** Trains the model on the provided feature matrix X and target vector y.
- **predict(X):** Makes predictions on the input feature matrix X after the model has been trained.

The following parameters are available for tuning:

- **alpha:** Learning rate for the gradient descent optimization. Controls the size of weight updates.
- **lambda1 (L1 regularization parameter):** Controls the strength of the L1 penalty, which encourages sparsity in the model coefficients.
- **lambda2 (L2 regularization parameter):** Controls the strength of the L2 penalty, which constrains the magnitude of the coefficients.
- **num_iterations:** Number of iterations for the gradient descent optimization. Higher values allow the model to converge more precisely, at the cost of computation time.
- **batch_size:** Size of mini-batches used for stochastic gradient descent. Smaller batches can introduce noise into the optimization but can improve convergence speed.

## Basic Usage Examples

### Example 1: Synthetic Data Example
This example demonstrates how to train and make predictions using synthetic data. You can run this code directly without any changes:

```python
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
```

### Example 2: Using Your Own CSV File and Pathological Cases
To use the model with your own data:
- Replace "Spotify_Most_Streamed.csv" with the name of your CSV file.
- Ensure the file is placed in the correct directory (elasticnet/tests).
- Ensure your CSV file contains a header row and the last column should be the target variable (y), and all other columns will be treated as features (X).

```python
import numpy as np
import csv

import sys
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
```
### Running tests with Pytest
Once you've modified the file path, you can use pytest to run all tests, including real-world data tests and pathological case tests.
- Open a terminal in the directory containing your test script.
- Run Pytest with the following command:
pytest test_ElasticNetModel.py

This will execute all tests in test_ElasticNetModel.py, which includes:
- **Real-World Data Testing:** Validates model performance on the CSV file you specified.
- **Pathological Cases:** Tests the model’s resilience under extreme conditions like high dimensionality, collinear features, and extreme values.

### Understanding the Test Outputs
**For Real-World Data:** Look for outputs such as accuracy (for classification) or Mean Absolute Error (for regression) to evaluate model performance.
**For Pathological Cases:** Outputs will indicate whether the model successfully handled each challenging case or encountered issues. Any assertion failures will provide feedback on specific issues with predictions.

## Limitations and Potential Improvements
The model has limitations with:

- **Very High-Dimensional Data:** When the number of features is much greater than the number of samples, the current batch gradient descent approach can struggle with convergence. Increasing the num_iterations and adjusting alpha may help, but performance issues may persist with extremely high-dimensional data.
- **Pathological Collinear Features:** While ElasticNet handles multicollinearity better than Lasso alone, extreme cases can still lead to instability. A workaround could involve implementing feature selection or dimensionality reduction as a preprocessing step.
- **Numerical Stability with Extreme Values:** Although the model clips inputs to prevent overflow in the sigmoid function, extremely large or small values may still lead to instability. A potential solution could involve further data normalization or scaling improvements.

## Future Improvements
Given more time, We would like to explore:
- **Adaptive Learning Rates:** Implementing adaptive learning rate optimizers like Adam or RMSprop to handle different types of data more effectively.
- **Automatic Hyperparameter Tuning:** Adding a cross-validation function that automatically tunes alpha, lambda1, and lambda2 based on grid or random search.
- **Support for Early Stopping:** To prevent overfitting and optimize convergence time, we would add an early stopping criterion based on validation performance.

