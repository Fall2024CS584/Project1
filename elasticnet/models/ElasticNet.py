import numpy as np

class ElasticNetModel:
    def __init__(self, regularization_strength, l1_ratio, max_iterations, tolerance=1e-6, learning_rate=0.01):
        """
        Initialize the ElasticNet regression model.

        Parameters:
        regularization_strength: Regularization strength (λ)
        l1_ratio: The mixing ratio between L1 and L2 (0 <= l1_ratio <= 1)
        max_iterations: Maximum number of iterations for gradient descent
        tolerance: Tolerance for stopping criterion
        learning_rate: Step size for gradient descent
        """
        self.reg_strength = regularization_strength
        self.l1_ratio = l1_ratio
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate

    def _soft_threshold(self, rho, l1_penalty):
        """Soft thresholding operator for L1 penalty."""
        if rho < -l1_penalty:
            return rho + l1_penalty
        elif rho > l1_penalty:
            return rho - l1_penalty
        else:
            return 0

    def _compute_loss(self, X, y, coefficients, intercept):
        """Compute the ElasticNet loss (MSE + L1 + L2 penalties)."""
        predictions = X.dot(coefficients) + intercept
        mse_loss = np.mean((y - predictions) ** 2)
        l1_penalty = self.l1_ratio * np.sum(np.abs(coefficients))
        l2_penalty = (1 - self.l1_ratio) * np.sum(coefficients ** 2)
        return mse_loss + self.reg_strength * (l1_penalty + l2_penalty)

    def fit(self, X, y):
        """
        Fit the model to the data using gradient descent.

        Parameters:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        """
        n_samples, n_features = X.shape

        # Normalize the features
        feature_mean = np.mean(X, axis=0)
        feature_std = np.std(X, axis=0)
        X = (X - feature_mean) / feature_std

        # Initialize coefficients and intercept
        coefficients = np.zeros(n_features)
        intercept = 0
        loss_history = []

        for iteration in range(self.max_iterations):
            predictions = X.dot(coefficients) + intercept
            residuals = predictions - y

            # Compute gradient for intercept
            intercept_gradient = np.sum(residuals) / n_samples
            intercept -= self.learning_rate * intercept_gradient

            # Compute gradient for coefficients (ElasticNet penalty)
            coef_gradient = X.T.dot(residuals) / n_samples + \
                            self.reg_strength * (self.l1_ratio * np.sign(coefficients) +
                                                 (1 - self.l1_ratio) * 2 * coefficients)

            # Update coefficients
            coefficients -= self.learning_rate * coef_gradient

            # Record the loss
            loss = self._compute_loss(X, y, coefficients, intercept)
            loss_history.append(loss)

            # Stopping condition (based on gradient tolerance)
            if np.linalg.norm(coef_gradient) < self.tolerance:
                break

        # Return the fitted model and results encapsulated in ElasticNetModelResults
        return ElasticNetModelResults(coefficients, intercept, feature_mean, feature_std, loss_history)

class ElasticNetModelResults:
    def __init__(self, coefficients, intercept, feature_mean, feature_std, loss_history):
        """
        Encapsulates the results of the ElasticNet model after fitting.

        Parameters:
        coefficients: Fitted coefficients for the model
        intercept: Fitted intercept for the model
        feature_mean: Mean of the features (used for normalization)
        feature_std: Standard deviation of the features (used for normalization)
        loss_history: History of the loss values during training
        """
        self.coefficients = coefficients
        self.intercept = intercept
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.loss_history = loss_history

    def predict(self, X):
        """
        Predict target values for given input features.

        Parameters:
        X: Feature matrix for which predictions are to be made

        Returns:
        predictions: Predicted target values
        """
        # Normalize the input data with the same scaling applied in fit
        X = (X - self.feature_mean) / self.feature_std
        return X.dot(self.coefficients) + self.intercept
    
    def r2_score(self, X, y_true):
        """
        Calculate the R-squared value for the model on given data.
        
        Parameters:
        X: Feature matrix
        y_true: Actual target values

        Returns:
        R² value
        """
        # Predict the values
        predictions = self.predict(X)
        
        # Total sum of squares (variance of the data)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        
        # Residual sum of squares (variance of the errors)
        ss_residual = np.sum((y_true - predictions) ** 2)
        
        # Compute R²
        r2 = 1 - (ss_residual / ss_total)
        return r2


    def print_summary(self):
        """
        Print a summary of the fitted model, including coefficients and intercept.
        """
        print("Model Summary:")
        print(f"Intercept: {self.intercept}")
        print(f"Coefficients: {self.coefficients}")
        print(f"Number of iterations: {len(self.loss_history)}")
        print(f"Final loss: {self.loss_history[-1]}" if self.loss_history else "No loss recorded.")
