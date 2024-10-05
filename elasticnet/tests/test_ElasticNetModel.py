import csv
import numpy as np
from elasticnet.models.ElasticNet import ElasticNetModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def test_predict():
    model = ElasticNetModel()
    data = []

    # Correct path to the CSV file
    csv_file_path = "elasticnet/tests/small_test.csv"

    # Reading the CSV data
    with open(csv_file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Extracting features (X) and target (y)
    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])

    # Split the data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Fitting the model on training data
    results = model.fit(X_train, y_train)

    # Making predictions on the test data
    preds = results.predict(X_test)

    # Calculate the Mean Squared Error and R-squared for the test set
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Print the MSE, R², coefficients, and intercept
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")
    print(f"Coefficients (beta): {results.coef_}")
    print(f"Intercept (beta0): {results.intercept_}")

    # Print the number of non-zero coefficients
    non_zero_coefs = np.sum(results.coef_ != 0)
    print(f"Number of non-zero coefficients: {non_zero_coefs}")

    # Dummy assertion, replace with actual evaluation
    assert isinstance(preds, np.ndarray), "Prediction is not an array"
