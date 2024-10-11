import numpy as np
import matplotlib.pyplot as plt  # type: ignore

class RegularizedRegression:
    def __init__(self, regularization='none', lambda_l1=0.0, lambda_l2=0.0, alpha=0.001, num_iterations=1000):
        self.regularization = regularization
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.parameters = None

    def linear_loss(self, parameters, features, labels):
        N = len(labels)
        predictions = np.dot(features, parameters)
        loss = np.sum((predictions - labels) ** 2) / (2 * N)
        return loss

    def linear_gradient(self, parameters, features, labels):
        N = len(labels)
        predictions = np.dot(features, parameters)
        grad = (1 / N) * np.dot(features.T, (predictions - labels))
        return grad

    def fit(self, X, y):
    
        initial_parameters = np.zeros(X.shape[1])
    
        if self.regularization == 'lasso':
            loss_function = self.lasso_loss
            gradient_function = self.lasso_gradient
        elif self.regularization == 'ridge':
            loss_function = self.ridge_loss
            gradient_function = self.ridge_gradient
        elif self.regularization == 'elastic_net':
            loss_function = self.elastic_net_loss
            gradient_function = self.elastic_net_gradient
        else:
            loss_function = self.linear_loss
            gradient_function = self.linear_gradient
        
        self.parameters = initial_parameters.copy()
        iteration_list, loss_list = [], []
        
        for i in range(self.num_iterations):
            grad = gradient_function(self.parameters, X, y)
            self.parameters -= self.alpha * grad
            loss = loss_function(self.parameters, X, y)
            iteration_list.append(i)
            loss_list.append(loss)


        # plt.plot(iteration_list, loss_list, 'ob', linestyle='solid', color='red')
        # plt.xlabel("Iterations")
        # plt.ylabel("Loss")
        # plt.title(f"Loss Function ({self.regularization.capitalize()})")
        # plt.show()

    def predict(self, X):
        return np.dot(X, self.parameters)
