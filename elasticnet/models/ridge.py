import numpy as np
from models.linear_regression import RegularizedRegression

class RidgeModel(RegularizedRegression):
    def __init__(self, lambda_l2=0.01, alpha=0.001, num_iterations=1000):
        super().__init__(regularization='ridge', lambda_l2=lambda_l2, alpha=alpha, num_iterations=num_iterations)

    def ridge_loss(self, parameters, features, labels):
        return self.linear_loss(parameters, features, labels) + self.lambda_l2 * np.linalg.norm(parameters) ** 2

    def ridge_gradient(self, parameters, features, labels):
        grad = self.linear_gradient(parameters, features, labels)
        grad += 2 * self.lambda_l2 * parameters
        return grad
    def predict(self, X):
        return np.dot(X, self.parameters)
