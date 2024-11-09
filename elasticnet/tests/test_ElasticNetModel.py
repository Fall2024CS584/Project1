import sys
import csv
import numpy as np
import os
'''
import io
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
'''
from elasticnet.models.ElasticNet import ElasticNetModel
from generate_positive_regression_data import generate_rotated_positive_data
from generate_nagetive_regression_data import generate_negative_data
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
# Original test code part
def test_predict_with_csv():
    # Initialize the ElasticNet model with the given parameters
    model = ElasticNetModel(lambdas=0.1, thresh=0.6, max_iter=1000, tol=1e-4, learning_rate=0.01)
    
    # Read the CSV file and load data
    data = []
    #with open("elasticnet/tests/small_test.csv", "r") as file:
    with open("small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Extract features (X) and target (y) from the data
    X = np.array([[v for k, v in datum.items() if k.startswith('x')] for datum in data], dtype=float)
    y = np.array([float(datum['y']) for datum in data], dtype=float)  # Convert 'y' to a 1D array

    # Split the data into 80% training and 20% testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model on the training data
    results = model.fit(X_train, y_train)

    # Make predictions on the test data
    preds = results.predict(X_test)

    # Calculate various evaluation metrics
    mse = np.mean((y_test - preds) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mae = mean_absolute_error(y_test, preds)  # Mean Absolute Error
    r2 = r2_score(y_test, preds)  # R-squared

    # Print the results of the evaluation metrics
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")


def test_with_generated_data(data_type = 0):
    if(data_type == 1):
        # Generate training data
        X, y = generate_negative_data(range_x=[-40, 40], noise_scale=5, size=300, num_features=6, seed=42)
        # Generate test dataset
        X_test, y_test = generate_negative_data(range_x=[-40, 40], noise_scale=50, size=50, num_features=6, seed=526)
        # Instantiate the ElasticNetModel and train
        model = model = ElasticNetModel(lambdas=0.1, thresh=0.5, max_iter=1000, tol=1e-4, learning_rate=0.0001)#must turn down the learning rate
    else:
        X, y = generate_rotated_positive_data(range_x=[-30, 30], noise_scale=2, size=200, num_features=6, seed=42, rotation_angle=45, mode=0)

        X_test, y_test = generate_rotated_positive_data(range_x=[-30, 30], noise_scale=50, size=50, num_features=6, seed=100, rotation_angle=45, mode=1)

        model = ElasticNetModel(lambdas=0.1, thresh=0.5, max_iter=1000, tol=1e-4, learning_rate=0.01)
    
    # Visualize the generated data
    plt.figure(figsize=(15, 6))
    for i in range(X.shape[1]):
        plt.subplot(1, X.shape[1], i + 1)
        plt.scatter(X[:, i], y, label=f'Feature {i + 1}', alpha=0.6)
        plt.xlabel(f'Feature {i + 1}')
        plt.ylabel('Y_Train')
        plt.title(f'Train Feature {i + 1}')
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Visualize the test data
    plt.figure(figsize=(15, 6))
    for i in range(X_test.shape[1]):
        plt.subplot(1, X_test.shape[1], i + 1)
        plt.scatter(X_test[:, i], y_test, label=f'Test Feature {i + 1}', alpha=0.6, color='orange')
        plt.xlabel(f'Feature {i + 1}')
        plt.ylabel('Y_test')
        plt.title(f'Test Feature {i + 1}')
        plt.grid(True)
    plt.tight_layout()
    plt.show()


    model_results = model.fit(X, y)
    # Predict on the test data
    y_pred = model_results.predict(X_test)

    # Visualize actual and predicted values
    for i in range(X_test.shape[1]):
        plt.subplot(2, 3, i + 1)
        plt.scatter(X_test[:, i], y_test, color='orange', alpha=0.6, label='Actual Values')
        plt.scatter(X_test[:, i], y_pred, color='blue', alpha=0.6, label='Predicted Values')
        
        # Fit line
        z = np.polyfit(X_test[:, i], y_test, 1)
        p = np.poly1d(z)
        sorted_indices = np.argsort(X_test[:, i])
        X_sorted = X_test[:, i][sorted_indices]
        y_fit = p(X_sorted)
        plt.plot(X_sorted, y_fit, "r--", label='Fit Line')

        plt.xlabel(f'Feature {i + 1}')
        plt.ylabel('Target')
        plt.title(f'Feature {i + 1} vs. Actual and Predicted Target')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

    # Manually calculate MSE
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

# Run the tests
if __name__ == "__main__":
    test_predict_with_csv()  # Test the original CSV code
    test_with_generated_data(data_type=0)  # Test with generated positive regression data
    test_with_generated_data(data_type=1) # Test with generated half positive and half nagetive regression data
