import numpy as np

class ElasticNet:
    def __init__(self, alpha=0.1, l1_ratio=0.5, learn_rate=0.01, max_iters=1000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learn_rate = learn_rate
        self.max_iters = max_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])  
        self.bias = 0 

        for i in range(self.max_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            error = y_pred - y

            dw = np.dot(X.T, error) / len(y) + self.alpha * (self.l1_ratio * np.sign(self.weights) + (1 - self.l1_ratio) * self.weights)
            db = np.mean(error)  

            self.weights -= self.learn_rate * dw
            self.bias -= self.learn_rate * db

            if i % 100 == 0:
                loss = self.calculate_loss(X, y)
                print(f"Iteration {i}, Loss: {loss:.4f}")

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def calculate_loss(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        l1 = self.l1_ratio * np.sum(np.abs(self.weights))
        l2 = (1 - self.l1_ratio) * np.sum(self.weights ** 2)
        return mse + self.alpha * (l1 + 0.5 * l2)

