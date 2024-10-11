import numpy as np

class ElasticNetModel():
    def __init__(self, alpha=1.0, l1_ratio=0.5, learning_rate=0.01, max_iter=1000, tol=1e-4):
        if not isinstance(alpha, (float, int)) or alpha < 0:
            raise ValueError("alpha must be a non-negative float or integer.")
        if not isinstance(l1_ratio, (float, int)) or not (0 <= l1_ratio <= 1):
            raise ValueError("l1_ratio must be a float between 0 and 1.")
        if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive float.")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(tol, (float, int)) or tol <= 0:
            raise ValueError("tol must be a positive float.")
        
        self.alpha = alpha  # Combined regularization strength
        self.l1_ratio = l1_ratio  # L1:L2 ratio (0 for Ridge, 1 for Lasso, between for ElasticNet)
        self.learning_rate = learning_rate  # Step size for gradient descent
        self.max_iter = max_iter  # Maximum iterations for gradient descent
        self.tol = tol  # Tolerance for stopping criterion
        self.weight_ = None
        self.bias_ = None
        self.is_fitted = False

        # L1 and L2 penalties
        self.l1_penalty = self.alpha * self.l1_ratio
        self.l2_penalty = self.alpha * (1 - self.l1_ratio)
        
    def _validate_input(self, X, y=None):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError("y must be a numpy array")
            if len(X) != len(y):
                raise ValueError("X and y must have the same number of samples")

        if np.isnan(X).any():
            raise ValueError("X contains NaN values")
        
        if y is not None and np.isnan(y).any():
            raise ValueError("y contains NaN values")
    
    def fit(self, X, y):
        self._validate_input(X, y)
        
        n_samples, n_features = X.shape
        self.weight_ = np.zeros(n_features)  # Initialize weights
        self.bias_ = 0  # Initialize bias
        
        self.is_fitted = True

        for _ in range(self.max_iter):
            y_pred = np.dot(X, self.weight_) + self.bias_  
            
            if np.isnan(y_pred).any():
                raise ValueError("NaN values detected in predictions during gradient descent")
            
            residuals = y - y_pred

            # Gradient calculation based on the formula you provided
            dW = np.zeros(n_features)  # Initialize gradient for weights
            for j in range(n_features):
                if self.weight_[j] > 0:
                    dW[j] = (-(2 * (X[:, j].dot(residuals)) + self.l1_penalty + 
                               2 * self.l2_penalty * self.weight_[j]) / n_samples)
                else:
                    dW[j] = (-(2 * (X[:, j].dot(residuals)) - self.l1_penalty + 
                               2 * self.l2_penalty * self.weight_[j]) / n_samples)
            
            # Gradient for bias
            db = -2 * np.sum(residuals) / n_samples  

            # Update weights and bias
            self.weight_ -= self.learning_rate * dW
            self.bias_ -= self.learning_rate * db

            # Check stopping criterion
            if np.linalg.norm(self.learning_rate * dW) < self.tol and abs(self.learning_rate * db) < self.tol:
                break
        
        return ElasticNetModelResults(self.weight_, self.bias_)

class ElasticNetModelResults():
    def __init__(self, weight_, bias_):
        self.weight_ = weight_
        self.bias_ = bias_

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        
        if np.isnan(X).any():
            raise ValueError("X contains NaN values")
        
        return np.dot(X, self.weight_) + self.bias_