import numpy as np

class ElasticNetModel():
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0

    def _soft_threshold(self, rho, lam):
        return np.sign(rho) * np.maximum(np.abs(rho) - lam, 0.0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)

        # Center X and y
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Initialize intercept
        self.intercept_ = y_mean

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            for j in range(n_features):
                # Residual excluding feature j
                residual = y_centered - (X_centered @ self.coef_ - X_centered[:, j] * self.coef_[j])
                rho = X_centered[:, j].T @ residual

                # Update coefficient j
                z = rho / n_samples
                denom = (np.sum(X_centered[:, j] ** 2) / n_samples) + self.alpha * (1 - self.l1_ratio)

                self.coef_[j] = self._soft_threshold(z, self.alpha * self.l1_ratio) / denom

            # Check convergence
            if np.linalg.norm(self.coef_ - coef_old, ord=2) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

class ElasticNetModelResults():
    def __init__(self, coef_, intercept_):
        self.coef_ = coef_
        self.intercept_ = intercept_

