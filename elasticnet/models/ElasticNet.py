import numpy as np


class ElasticNetModel:
    def __init__(self, alpha=1, l1_ratio=0.5, max_iter=1000, tol=1e-5, learning_rate=0.1):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Standardize the features
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        non_zero_std = X_std != 0  # Detect non-zero variance features
        X_std[~non_zero_std] = 1  # Prevent division by zero

        X_scaled = (X - X_mean) / X_std

        # Center the target
        y_mean = y.mean()
        y_centered = y - y_mean

        n_samples, n_features = X_scaled.shape
        coef_ = np.zeros(n_features)
        intercept_ = y_mean

        # Gradient descent loop
        for iteration in range(self.max_iter):
            y_pred = X_scaled @ coef_ + intercept_

            # Compute the residuals
            residuals = y_pred - y_centered

            # Compute the gradients (grad_coef should be of shape (n_features,))
            grad_coef = (X_scaled.T @ residuals) / n_samples

            # Add regularization terms
            regularization_term = self.alpha * (self.l1_ratio * np.sign(coef_) + (1 - self.l1_ratio) * coef_)
            grad_coef += regularization_term

            grad_intercept = np.sum(residuals) / n_samples

            # Update the coefficients and intercept
            coef_ -= self.learning_rate * grad_coef
            intercept_ -= self.learning_rate * grad_intercept

            # Check convergence
            coef_change = np.linalg.norm(grad_coef, ord=2)
            if coef_change < self.tol:
                break

        # Unscale the coefficients to match the original feature scale
        self.coef_ = coef_ / X_std
        self.intercept_ = y_mean - X_mean @ self.coef_

        return ElasticNetModelResults(self.coef_, self.intercept_)


class ElasticNetModelResults:
    def __init__(self, coef_, intercept_):
        self.coef_ = coef_
        self.intercept_ = intercept_

    def predict(self, X):
        """Predicts the target values using the fitted model parameters."""
        return X @ self.coef_ + self.intercept_


class MyMSE:
    @staticmethod
    def calculate(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


class MyRSquared:
    @staticmethod
    def calculate(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)