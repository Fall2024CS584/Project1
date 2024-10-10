
class ElasticNetModel:
    
    def __init__(self, lambdas=1.0, l1_ratio=0.5, iterations=10000, learning_rate=0.001):
        self.lambdas = lambdas
        self.l1_ratio = l1_ratio
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = 0

    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        # Performing gradient descent
        for _ in range(self.iterations):
            current_predictions = np.dot(X, self.coef_) + self.intercept_
            residuals = current_predictions - y
        
            # Computing gradients for coefficients
            # First, we calculate the gradient from the residuals
            residual_gradient = np.dot(X.T, residuals) / n_samples
        
            # Computing the L1 regularization term
            l1_term = self.l1_ratio * self.lambdas * np.sign(self.coef_)
        
            # Computing the L2 regularization term
            l2_term = (1 - self.l1_ratio) * self.lambdas * 2 * self.coef_
        
            # Combining the gradients from residuals, L1, and L2 terms
            coef_gradient = residual_gradient + l1_term + l2_term
        
            # Computing the gradient for the intercept
            intercept_gradient = np.sum(residuals) / n_samples

            # Updating the model parameters
            self.coef_ -= self.learning_rate * coef_gradient
            self.intercept_ -= self.learning_rate * intercept_gradient

        return ElasticNetModelResults(self.coef_, self.intercept_)

class ElasticNetModelResults:
    def __init__(self, coef, intercept):
        self.coef_ = coef
        self.intercept_ = intercept

    def predict(self, X):
      
        return np.dot(X, self.coef_) + self.intercept_
