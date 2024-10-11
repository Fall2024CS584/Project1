import numpy as np
from models.linear_regression import RegularizedRegression

class ElasticNetModel(RegularizedRegression):
    def __init__(self, lambda_l1=0.01, lambda_l2=0.01, alpha=0.001, num_iterations=1000):
        super().__init__(regularization='elastic_net', lambda_l1=lambda_l1, lambda_l2=lambda_l2, alpha=alpha, num_iterations=num_iterations)
        self.parameters = None 

    def elastic_net_loss(self, parameters, features, labels):
        mse_loss = self.linear_loss(parameters, features, labels)
        l1_loss = self.lambda_l1 * np.sum(np.abs(parameters))
        l2_loss = self.lambda_l2 * np.linalg.norm(parameters) ** 2
        return mse_loss + l1_loss + l2_loss

    def elastic_net_gradient(self, parameters, features, labels):
        grad = self.linear_gradient(parameters, features, labels)
        l1_grad = self.lambda_l1 * np.sign(parameters)
        l2_grad = 2 * self.lambda_l2 * parameters
        return grad + l1_grad + l2_grad

    

    def predict(self, X):
        return np.dot(X, self.parameters)

class ElasticNetModelResults:
    def __init__(self, parameters):
        self.parameters = parameters

    def predict(self, x):
        return np.dot(x, self.parameters)
