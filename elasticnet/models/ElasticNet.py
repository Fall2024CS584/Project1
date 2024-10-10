import numpy as np

class ElasticNetModel():
    def __init__(self, alpha=0.1, l1_ratio=0.5, learning_rate=0.01, iterations=1000):
        # We are setting the hyperparamenters for the ElasticNet model 
        self.alpha = alpha  # Alphs --> Strength of the regularization
        self.l1_ratio = l1_ratio  
        self.learning_rate = learning_rate  # Learning rate for the gradient descent
        self.iterations = iterations  # Number of iterations for the gradient descent

    def initialize_weights(self, n_features):
        # Initializing all the weights and bias.
        self.weights = np.zeros(n_features)
        self.bias = 0

    def predict(self, X):
        # Predicting the values based on the current weights and the bias
        return np.dot(X, self.weights) + self.bias

    def compute_cost(self, X, y):
        # Computing the cost along with the ElasticNet regularization.
        m = len(y)
        predictions = self.predict(X)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)  # MSE
        
        # Adding the ElasticNet regularization this is the combination of both(L1 + L2)
        l1_penalty = self.l1_ratio * np.sum(np.abs(self.weights))
        l2_penalty = (1 - self.l1_ratio) * np.sum(self.weights ** 2)
        regularization = self.alpha * (l1_penalty + l2_penalty / 2)
        
        return cost + regularization

    def gradient_descent(self, X, y):
        # Performing the gradient descent to the updating of weights and bias.
        m = len(y)
        
        for i in range(self.iterations):
            predictions = self.predict(X)
            
            # Calculaing the gradients for weights and bias
            dw = (1 / m) * np.dot(X.T, (predictions - y)) + self.alpha * (self.l1_ratio * np.sign(self.weights) + (1 - self.l1_ratio) * self.weights)
            db = (1 / m) * np.sum(predictions - y)
            
            # Updating 
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 100 == 0:
                cost = self.compute_cost(X, y)
                print(f"Iteration {i}, Cost: {cost}")
    
    def fit(self, X, y):
        # Training my ElasticNet model using the gradient descent.
        n_features = X.shape[1]
        self.initialize_weights(n_features)
        
        self.gradient_descent(X, y)
        
        return ElasticNetModelResults(self.weights, self.bias)


class ElasticNetModelResults():
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
