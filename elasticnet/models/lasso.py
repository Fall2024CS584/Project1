import numpy as np
from models.linear_regression import RegularizedRegression

class LassoModel(RegularizedRegression):
    def __init__(self, lambda_l1=0.01, alpha=0.001, num_iterations=1000):
        super().__init__(regularization='lasso', lambda_l1=lambda_l1, alpha=alpha, num_iterations=num_iterations)

    def lasso_loss(self, parameters, features, labels):
        return self.linear_loss(parameters, features, labels) + self.lambda_l1 * np.sum(np.abs(parameters))

    def lasso_gradient(self, parameters, features, labels):
        grad = self.linear_gradient(parameters, features, labels)
        grad += self.lambda_l1 * np.sign(parameters)
        return grad
    def predict(self, X):
        return np.dot(X, self.parameters)
