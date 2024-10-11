import csv
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from elasticnet.models.ElasticNet import ElasticNetModel
import matplotlib.pyplot as plt

def preprocess_data(data):
    # Convert 'sex' and 'child' columns to numeric
    for row in data:
        row['sex'] = 0 if row['sex'] == 'male' else 1
        row['child'] = 0 if row['child'] == 'no' else 1

    # Prepare X (features) and y (target 'rate')
    X = np.array([[float(row[k]) for k in row.keys() if k != 'rate'] for row in data])
    y = np.array([float(row['rate']) for row in data])

    return X, y

def test_predict():
    model = ElasticNetModel()
    data = []

    # Load data from test.csv
    with open("elasticnet/tests/test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Preprocess data to handle categorical values
    X, y = preprocess_data(data)

    # Print the fields (column names) being used for training
    print("Fields used for training:", list(data[0].keys())[:-1])  # Excluding the target ('rate')

    # Fit the model and make predictions
    results = model.fit(X, y)
    preds = results.predict(X)

    # Print all training data and predictions
    print("\n---- Training Data (all rows) ----")
    print(X)
    print("\n---- Predicted Values (all rows) ----")
    print(preds)

    # Calculate and print evaluation metrics
    mse = mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print("\n---- Evaluation Metrics ----")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R²): {r2}")

    # Visualizations
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Actual vs Predicted
    axs[0, 0].scatter(y, preds, color='blue', alpha=0.6)
    axs[0, 0].plot([min(y), max(y)], [min(y), max(y)], color='red', linewidth=2)
    axs[0, 0].set_title("Actual vs Predicted")
    axs[0, 0].set_xlabel('Actual')
    axs[0, 0].set_ylabel('Predicted')

    # Residuals
    residuals = y - preds
    axs[0, 1].scatter(preds, residuals, color='purple', alpha=0.6)
    axs[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axs[0, 1].set_title("Residuals")
    axs[0, 1].set_xlabel('Predicted')
    axs[0, 1].set_ylabel('Residuals')

    # Distribution of Target Values
    axs[1, 0].hist(y, bins=50, color='green', alpha=0.7)
    axs[1, 0].set_title("Distribution of 'rate' (Target)")
    axs[1, 0].set_xlabel('rate')
    axs[1, 0].set_ylabel('Frequency')

    # Feature Weights
    axs[1, 1].bar(range(len(model.weights)), model.weights, color='orange')
    axs[1, 1].set_title("Feature Weights")
    axs[1, 1].set_xlabel('Feature Index')
    axs[1, 1].set_ylabel('Weight')

    plt.tight_layout(pad=5.0)
    plt.show()

if __name__ == "__main__":
    test_predict()
