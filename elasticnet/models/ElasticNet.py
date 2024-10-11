import numpy as np

class ElasticNetModel:
    def __init__(self, l1_penalty=1.0, l2_penalty=1.0, learning_rate=0.001, max_iterations=1000, tolerance=1e-5):
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for i in range(self.max_iterations):
            predictions = self.predict(X)
            errors = predictions - y

            gradient_weights = (X.T @ errors) / num_samples + self.l1_penalty * np.sign(self.weights) + self.l2_penalty * self.weights
            gradient_bias = np.mean(errors)

            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

            if np.all(np.abs(gradient_weights) < self.tolerance):
                break

        return ElasticNetModelResults(self.weights, self.bias)

    def predict(self, X):
        return X @ self.weights + self.bias


class ElasticNetModelResults:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, X):
        return X @ self.weights + self.bias
