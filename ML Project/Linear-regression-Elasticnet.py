import numpy as np

class ElasticNetLinearRegression:
    def __init__(self, alpha=1.0, l1_ratio=0.5, learning_rate=0.01, iterations=1000):
        """
        Elastic Net Linear Regression constructor.

        Parameters:
        alpha : Regularization strength (alpha > 0). Higher alpha increases regularization.
        l1_ratio : The ratio between L1 (Lasso) and L2 (Ridge) regularization.
                   l1_ratio = 0 corresponds to L2 penalty (Ridge),
                   l1_ratio = 1 corresponds to L1 penalty (Lasso),
                   and 0 < l1_ratio < 1 corresponds to Elastic Net.
        learning_rate : The step size for gradient descent. Controls how much to adjust the model per iteration.
        iterations : Number of iterations for the gradient descent optimization.
        """
        self.alpha = alpha  # Store regularization strength
        self.l1_ratio = l1_ratio  # Store ratio between L1 and L2 regularization
        self.learning_rate = learning_rate  # Store learning rate for gradient descent
        self.iterations = iterations  # Store number of iterations for gradient descent

    def fit(self, X, y):
        """
        Fit the model using gradient descent to minimize the cost function.

        Parameters:
        X : Input feature matrix (num_samples x num_features)
        y : Target vector (num_samples,)
        """
        self.m, self.n = X.shape  # Number of samples (m) and number of features (n)
        self.theta = np.zeros(self.n)  # Initialize weights (coefficients) to zeros
        self.bias = 0  # Initialize bias (intercept) to zero

        # Gradient descent optimization loop
        for _ in range(self.iterations):
            y_pred = self.predict(X)  # Compute predictions based on current weights and bias

            # Compute gradients for weights (theta) using Elastic Net regularization
            d_theta = (1 / self.m) * (X.T @ (y_pred - y)) + self.alpha * (
                self.l1_ratio * np.sign(self.theta) +  # L1 penalty (Lasso)
                (1 - self.l1_ratio) * self.theta)  # L2 penalty (Ridge)

            # Compute gradient for bias (intercept)
            d_bias = (1 / self.m) * np.sum(y_pred - y)

            # Update weights and bias using the gradients
            self.theta -= self.learning_rate * d_theta  # Update weights
            self.bias -= self.learning_rate * d_bias  # Update bias

    def predict(self, X):
        """
        Make predictions using the learned weights and bias.

        Parameters:
        X : Input feature matrix (num_samples x num_features)

        Returns:
        y_pred : Predictions (num_samples,)
        """
        return X @ self.theta + self.bias  # Linear prediction (X * theta + bias)

    def mse(self, y_true, y_pred):
        """
        Compute Mean Squared Error (MSE) as a performance metric.

        Parameters:
        y_true : True target values (ground truth)
        y_pred : Predicted values from the model

        Returns:
        mse : Mean Squared Error, a measure of how close the predictions are to the true values.
        """
        return np.mean((y_true - y_pred) ** 2)  # Calculate the average of the squared differences


# Sample data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([6, 8, 9, 11])

# Initialize model
model = ElasticNetLinearRegression(alpha=0.1, l1_ratio=0.5, learning_rate=0.01, iterations=1000)

# Fit the model to data
model.fit(X, y)

# Predict using the model
y_pred = model.predict(X)

# Print predictions and the MSE
print("Predictions:", y_pred)
print("MSE:", model.mse(y, y_pred))