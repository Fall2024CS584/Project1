class ElasticNetModel:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, epochs=1):
        self.alpha = alpha  # Regularization strength
        self.l1_ratio = l1_ratio  # Balance between L1 and L2 penalties
        self.max_iter = max_iter  # Maximum number of iterations per epoch
        self.tol = tol  # Tolerance for convergence
        self.epochs = epochs  # Number of passes over the dataset
        self.coef_ = None  # Coefficients after training
        self.intercept_ = 0.0  # Intercept

    def _soft_thresholding(self, rho, lambda_):
        """Apply the soft-thresholding operator for L1 penalty."""
        if rho < -lambda_:
            return rho + lambda_
        elif rho > lambda_:
            return rho - lambda_
        else:
            return 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize coefficients
        self.coef_ = np.zeros(n_features)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")

            for iteration in range(self.max_iter):
                coef_old = np.copy(self.coef_)

                # Update each coefficient using coordinate descent
                for j in range(n_features):
                    # Compute the residual excluding feature j
                    residual = y - (X @ self.coef_) + self.coef_[j] * X[:, j]
                    rho = X[:, j].T @ residual

                    # Update the j-th coefficient using soft-thresholding
                    weight_update = self._soft_thresholding(rho, self.alpha * self.l1_ratio)
                    self.coef_[j] = weight_update / (X[:, j].T @ X[:, j] + self.alpha * (1 - self.l1_ratio))

                # Check convergence after weight update
                if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                    print(f"Convergence reached at iteration {iteration+1}")
                    break

            print(f"Weights after epoch {epoch+1}: {self.coef_}")

        # Calculate intercept as the mean difference between actual and predicted values
        self.intercept_ = np.mean(y - X @ self.coef_)

        return ElasticNetModelResults(self.coef_, self.intercept_)

class ElasticNetModelResults:
    def __init__(self, coef_, intercept_):
        self.coef_ = coef_
        self.intercept_ = intercept_

    def predict(self, X):
        """Make predictions using the learned coefficients."""
        return X @ self.coef_ + self.intercept_

    def evaluate(self, X, y_true):
      """Evaluate the model performance and return MSE and R-squared metrics."""
      y_pred = self.predict(X)
      mse = mean_squared_error(y_true, y_pred)
      r2 = r2_score(y_true, y_pred)
      return mse, r2

def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / n
    return mse

def r2_score(y_true, y_pred):
    total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
    explained_variance = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (explained_variance / total_variance)
    return r2
