import numpy as np


class ElasticNetModel:
    def __init__(self, alpha=1.0, l1_ratio=0.5, tol=1e-4, max_iter=1000):
        """
        ElasticNet linear regression model.

        Parameters:
        alpha (float): Regularization strength.
        l1_ratio (float): The mixing parameter between L1 and L2 (0 <= l1_ratio <= 1).
                          l1_ratio = 1 corresponds to Lasso, l1_ratio = 0 to Ridge.
        tol (float): Tolerance for stopping criteria.
        max_iter (int): Maximum number of iterations.
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = 0

    def _soft_thresholding(self, rho, lambda_):
        """Helper function for soft-thresholding."""
        if rho < -lambda_:
            return (rho + lambda_)
        elif rho > lambda_:
            return (rho - lambda_)
        else:
            return 0

    def fit(self, X, y):
        """Fit ElasticNet model to the data."""
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)

        # Coordinate Descent Optimization
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()

            for j in range(n_features):
                # Predict without feature j
                y_pred = X @ self.coef_
                residual = y - y_pred + X[:, j] * self.coef_[j]
                rho_j = np.dot(X[:, j], residual)

                # Update coefficient using soft-thresholding
                self.coef_[j] = self._soft_thresholding(rho_j, self.alpha * self.l1_ratio) / (
                            np.dot(X[:, j], X[:, j]) + self.alpha * (1 - self.l1_ratio))

            # Check for convergence
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                break

        return ElasticNetModelResults(self.coef_, self.intercept_)


class ElasticNetModelResults:
    def __init__(self, coef, intercept):
        self.coef_ = coef
        self.intercept_ = intercept

    def predict(self, X):
        """Predict using the fitted ElasticNet model."""
        return np.dot(X, self.coef_) + self.intercept_