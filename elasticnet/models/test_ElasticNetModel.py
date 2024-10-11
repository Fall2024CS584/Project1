import csv
import numpy as np
import matplotlib.pyplot as plt
from ElasticNet import ElasticNetModel
from metrics import mean_squared_error, mean_absolute_error

# Load the data from a CSV file
def load_data(filename):
    data = []
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])
    return X, y

# Plot predictions vs actual values for regression
def plot_predictions(y_true, y_pred, filename='predictions.png'):
    plt.figure(figsize=(8, 6))
    colors = np.abs(y_true - y_pred)
    plt.scatter(y_true, y_pred, c=colors, cmap='viridis', alpha=0.7)
    plt.colorbar(label="Difference (|y_true - y_pred|)")
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.savefig(filename)
    plt.close()

# Plot residuals for regression
def plot_residuals(y_true, y_pred, filename='residuals.png'):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    colors = residuals
    plt.scatter(y_pred, residuals, c=colors, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label="Residuals")
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.axhline(0, color='r', linestyle='--')
    plt.savefig(filename)
    plt.close()

# Plot classification decision boundaries
def plot_classification(X, y, model, filename='classification.png'):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Predict class for each point in mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')

    # Scatter plot
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap='viridis')
    plt.colorbar(scatter, label="Class")
    plt.title("Classification with ElasticNet")
    plt.savefig(filename)
    plt.close()

# Test the ElasticNet model's prediction
def test_predict(task='regression'):
    # Initialize the model with values
    model = ElasticNetModel(alpha=0.1, rho=0.5, max_iter=1000, tol=1e-4)
    
    # Load data
    X, y = load_data("/Users/gurjotsinghkalsi/Desktop/Fall2024/MachineLearning/Project1/elasticnet/models/small_test.csv")
    
    # Fit the model and get predictions
    results = model.fit(X, y)
    preds = results.predict(X)
    
    if task == 'regression':
        # Calculate metrics for regression
        mse = mean_squared_error(y, preds)
        mae = mean_absolute_error(y, preds)
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        
        # Assert that MSE and MAE are below certain thresholds
        assert mse < 1.0, f"MSE too high: {mse}"
        assert mae < 1.0, f"MAE too high: {mae}"
        
        # Plot regression results
        plot_predictions(y, preds, filename='/Users/gurjotsinghkalsi/Desktop/Fall2024/MachineLearning/Project1/elasticnet/models/predictions.png')
        plot_residuals(y, preds, filename='/Users/gurjotsinghkalsi/Desktop/Fall2024/MachineLearning/Project1/elasticnet/models/residuals.png')
        print("Regression plots saved: predictions.png and residuals.png")

if __name__ == "__main__":
    test_predict(task='regression')
