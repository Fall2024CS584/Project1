import numpy as np

class ElasticNetModel():
    def __init__(self,lambdas,thresh = 0.5,max_iter=1000, tol=1e-4,learning_rate=0.01):
        self.lambdas = lambdas
        self.thresh = thresh# between [0,1],0 means only L1 regularization, 1 means only L2 regularization      
        self.max_iter = max_iter#Maximum number of iterations
        self.tol = tol
        self.learning_rate = learning_rate       
        self.w = None
        self.intercept_ = None
        pass


    def fit(self, X, y):

        def calculate_weights(X, residual):
            gradient = (2 / X.shape[0]) * (X.T @ residual)  # MSE gradient
            l1_grad = self.lambdas * self.thresh * np.sign(self.w)  # L1 penalty gradient
            l2_grad = 2 * self.lambdas * (1 - self.thresh) * self.w  # L2 penalty gradien

            # make sure l1_grad and l2_grad has same dimention with gradient 
            if l1_grad.ndim == 1:
                l1_grad = l1_grad.reshape(-1)
            if l2_grad.ndim == 1:
                l2_grad = l2_grad.reshape(-1)

            if gradient.ndim > 1:
                gradient = np.sum(gradient, axis=1)

            gradient += l1_grad + l2_grad
            return gradient

        # Add intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.zeros(X.shape[1])  # Initialize weights including intercept
        Max_iter = self.max_iter

        while Max_iter > 0:
            Max_iter -= 1
            # Compute predictions and residuals
            pred = X @ self.w
            residual = pred - y

            # Calculate weight gradients
            gradient = calculate_weights(X, residual)
            # Update weights
            w_old = self.w.copy()
            self.w -= self.learning_rate * gradient

            # Check convergence
            if np.linalg.norm(self.w - w_old) < self.tol:
                break
        # divide intercept and coef
        self.intercept = self.w[0]
        self.coef = self.w[1:]
        return ElasticNetResults(self.coef, self.intercept)



class ElasticNetResults():
    def __init__(self, coef, intercept):
        self.coef = coef
        self.intercept = intercept
        pass

    def predict(self, X):
        return X @ self.coef + self.intercept
