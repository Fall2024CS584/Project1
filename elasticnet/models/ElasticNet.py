import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# ------------------- Utility Functions --------------------

def normalize_features(X):
    """Normalize the feature matrix to have zero mean and unit variance"""
    avg = np.mean(X, axis=0)
    deviation = np.std(X, axis=0)
    return (X - avg) / deviation, avg, deviation

def compute_performance(y_actual, y_predicted):
    """Compute Mean Squared Error and R-squared"""
    mse = np.mean((y_actual - y_predicted) ** 2)
    r_squared = 1 - (np.sum((y_actual - y_predicted) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2))
    residuals = y_actual - y_predicted
    return mse, r_squared, residuals

def show_comparison_plot(y_actual, y_predicted, title="Actual vs Predicted"):
    """Scatter plot for actual vs predicted values"""
    plt.scatter(y_actual, y_predicted, color='green', label='Predicted vs Actual')
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='red', label='Perfect Fit')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.show()

def show_residual_distribution(residuals, title="Residual Distribution"):
    """Histogram for residual distribution"""
    plt.hist(residuals, bins=15, color='orange', edgecolor='black')
    plt.title(title)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

def display_model_performance(model_name, mse, r_squared):
    """Print out the performance metrics of the model"""
    print(f"\nPerformance for {model_name}:")
    print(f"MSE: {mse:.3f}")
    print(f"R-squared: {r_squared:.3f}")
    print(f"Accuracy: {r_squared * 100:.2f}%")

# ------------------- Regression Implementations --------------------

def elastic_net_regression(X, y, alpha=1.0, l1_ratio=0.5):
    """Elastic Net regression from scratch"""
    num_samples, num_features = X.shape
    identity_matrix = np.eye(num_features)
    l1_term = l1_ratio * alpha
    l2_term = (1 - l1_ratio) * alpha
    coefficients = inv(X.T @ X + l2_term * identity_matrix) @ (X.T @ y - l1_term * np.sign(X.T @ y))
    return coefficients

def ridge_regression(X, y, alpha=1.0):
    """Implementing Ridge Regression from scratch"""
    num_samples, num_features = X.shape
    identity_matrix = np.eye(num_features)
    coefficients = inv(X.T @ X + alpha * identity_matrix) @ X.T @ y
    return coefficients

def lasso_regression(X, y, alpha=1.0):
    """Basic Lasso Regression implementation"""
    num_samples, num_features = X.shape
    coefficients = inv(X.T @ X) @ (X.T @ y - alpha * np.sign(X.T @ y))
    return coefficients

def polynomial_regression(X, y, degree=2):
    """Fit polynomial regression without external libraries"""
    poly_features = np.hstack([X ** i for i in range(1, degree + 1)])
    coefficients = inv(poly_features.T @ poly_features) @ poly_features.T @ y
    return coefficients, poly_features

# ------------------- Data Preparation --------------------

np.random.seed(42)
X = np.random.randn(100, 3)
y = np.dot(X, np.array([3, 1.5, -2])) + np.random.randn(100)

# Normalize features
X, X_avg, X_std = normalize_features(X)

# Train/Test split
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# ------------------- Model Training --------------------

# Elastic Net Model Training
elastic_net_coeff = elastic_net_regression(X_train, y_train, alpha=0.01, l1_ratio=0.9)
y_pred_enet = X_test @ elastic_net_coeff
mse_enet, r2_enet, res_enet = compute_performance(y_test, y_pred_enet)
display_model_performance("ElasticNet", mse_enet, r2_enet)

# Ridge Regression Model Training
ridge_coeff = ridge_regression(X_train, y_train, alpha=1.0)
y_pred_ridge = X_test @ ridge_coeff
mse_ridge, r2_ridge, res_ridge = compute_performance(y_test, y_pred_ridge)
display_model_performance("Ridge", mse_ridge, r2_ridge)

# Lasso Regression Model Training
lasso_coeff = lasso_regression(X_train, y_train, alpha=1.0)
y_pred_lasso = X_test @ lasso_coeff
mse_lasso, r2_lasso, res_lasso = compute_performance(y_test, y_pred_lasso)
display_model_performance("Lasso", mse_lasso, r2_lasso)

# Polynomial Regression Model Training
poly_coeff, X_train_poly = polynomial_regression(X_train, y_train, degree=2)
X_test_poly = np.hstack([X_test ** i for i in range(1, 3)])
y_pred_poly = X_test_poly @ poly_coeff
mse_poly, r2_poly, res_poly = compute_performance(y_test, y_pred_poly)
display_model_performance("Polynomial Regression", mse_poly, r2_poly)

# ------------------- Visualization --------------------

# Plotting predictions
show_comparison_plot(y_test, y_pred_enet, title="ElasticNet: Actual vs Predicted")
show_comparison_plot(y_test, y_pred_ridge, title="Ridge: Actual vs Predicted")
show_comparison_plot(y_test, y_pred_lasso, title="Lasso: Actual vs Predicted")
show_comparison_plot(y_test, y_pred_poly, title="Polynomial Regression: Actual vs Predicted")

# Plotting residual distributions
show_residual_distribution(res_enet, title="ElasticNet Residual Distribution")
show_residual_distribution(res_ridge, title="Ridge Residual Distribution")
show_residual_distribution(res_lasso, title="Lasso Residual Distribution")
show_residual_distribution(res_poly, title="Polynomial Regression Residual Distribution")

