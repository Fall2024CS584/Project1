import numpy as np
from sklearn.model_selection import train_test_split

class CustomElasticNet:
    def __init__(self, alpha=1.0, l1_ratio=0.5, tolerance=1e-4, max_iterations=1000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.coefficients_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Inserting a column for the intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        num_samples, num_features = X.shape
        
        # Initializing model coefficients in the below
        self.coefficients_ = np.zeros(num_features)
        
        for iteration in range(self.max_iterations):
            predictions = X @ self.coefficients_
            residuals = y - predictions
            
            # Updating model coefficients with ElasticNet regularization
            for j in range(num_features):
                if j == 0:  # Special handling for the intercept
                    gradient = -2 * np.sum(residuals) / num_samples
                    self.coefficients_[j] -= self.alpha * gradient
                else:
                    gradient = -2 * (X[:, j] @ residuals) / num_samples
                    l1_penalty = self.l1_ratio * self.alpha * np.sign(self.coefficients_[j])
                    l2_penalty = (1 - self.l1_ratio) * self.alpha * self.coefficients_[j]
                    self.coefficients_[j] -= self.alpha * (gradient + l1_penalty + l2_penalty)
            
            # Checking if the gradient is below the tolerance level for convergence
            if np.sum(np.abs(gradient)) < self.tolerance:
                break

        self.intercept_ = self.coefficients_[0]
        self.coefficients_ = self.coefficients_[1:]
        return ElasticNetResults(self.intercept_, self.coefficients_)

class ElasticNetResults:
    def __init__(self, intercept, coefficients):
        self.intercept_ = intercept
        self.coefficients_ = coefficients

    def predict(self, X):
        return self.intercept_ + X @ self.coefficients_

# Root Mean Squared Error (RMSE)
def rmse(y_actual, y_predicted):
    return np.sqrt(np.mean((y_actual - y_predicted) ** 2))

# Usage
if __name__ == "__main__":
    # Creating synthetic data
    X = np.random.rand(100, 3)
    y = 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + np.random.randn(100)

    # Spliting data for training (70%) and testing (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initializing and train the ElasticNet model
    model = CustomElasticNet(alpha=0.1, l1_ratio=0.7)
    results = model.fit(X_train, y_train)

    # Predicting on test data and evaluate performance
    y_pred = results.predict(X_test)
    test_rmse = rmse(y_test, y_pred)
    print("Test RMSE:", test_rmse)
