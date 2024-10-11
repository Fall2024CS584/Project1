import numpy as np

class ElasticNetModel:
    def __init__(self, alpha=1.0, rho=0.5, max_iter=1000, tol=1e-4):
        """
        Initialize the ElasticNet model.

        Parameters:
        - alpha: float, regularization strength.
        - rho: float, mixing parameter between L1 and L2 (0 <= rho <= 1).
        - max_iter: int, maximum number of iterations.
        - tol: float, tolerance for convergence.
        """
        self.alpha = alpha
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y):
        """
        Fit the ElasticNet model to the data.

        Parameters:
        - X: ndarray of shape (n_samples, n_features)
        - y: ndarray of shape (n_samples,)
        
        Returns:
        - ElasticNetModelResults: fitted model results containing intercept and coefficients.
        """
        # Standardize features
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # To avoid division by zero
        self.scale_[self.scale_ == 0] = 1
        X_std = (X - self.mean_) / self.scale_

        n_samples, n_features = X_std.shape
        X_aug = np.hstack((np.ones((n_samples, 1)), X_std))  # Add intercept
        n_features += 1  # Account for intercept

        self.coef_ = np.zeros(n_features)
        X_squared_sum = np.sum(X_aug ** 2, axis=0)

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            for j in range(n_features):
                # Compute residual excluding feature j
                residual = y - X_aug @ self.coef_ + self.coef_[j] * X_aug[:, j]

                if j == 0:
                    # Update intercept (no regularization)
                    self.coef_[j] = np.mean(residual)
                else:
                    # Compute rho_alpha and denominator with L2 term
                    rho_alpha = self.alpha * self.rho
                    denominator = X_squared_sum[j] + self.alpha * (1 - self.rho)

                    # Compute raw update
                    ro = np.dot(X_aug[:, j], residual)

                    # Apply soft-thresholding
                    if ro < -rho_alpha:
                        self.coef_[j] = (ro + rho_alpha) / denominator
                    elif ro > rho_alpha:
                        self.coef_[j] = (ro - rho_alpha) / denominator
                    else:
                        self.coef_[j] = 0.0

            # Check for convergence
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                print(f"Converged in {iteration + 1} iterations.")
                break
        else:
            print(f"Did not converge within {self.max_iter} iterations.")

        # Separate intercept and coefficients
        self.intercept_ = self.coef_[0] - np.sum((self.coef_[1:] * self.mean_) / self.scale_)
        self.coef_ = self.coef_[1:] / self.scale_
        return ElasticNetModelResults(self.intercept_, self.coef_)

    def predict(self, X):
        """
        Predict using the ElasticNet model.

        Parameters:
        - X: ndarray of shape (n_samples, n_features)

        Returns:
        - y_pred: ndarray of shape (n_samples,)
        """
        return X @ self.coef_ + self.intercept_

class ElasticNetModelResults:
    def __init__(self, intercept, coef):
        """
        Store the intercept and coefficients.

        Parameters:
        - intercept: float
        - coef: ndarray of shape (n_features,)
        """
        self.intercept_ = intercept
        self.coef_ = coef

    def predict(self, X):
        """
        Predict using the ElasticNet model.

        Parameters:
        - X: ndarray of shape (n_samples, n_features)

        Returns:
        - y_pred: ndarray of shape (n_samples,)
        """
        return X @ self.coef_ + self.intercept_