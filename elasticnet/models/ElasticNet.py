import numpy as np

class ElasticNetModel():
    def __init__(self, alpha=0.1, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.c = None
        self.b = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1, 1)
        i, j = X.shape
        self.c = np.zeros(j)
        self.b = 0.0

        for _ in range(10000):
            yp = self.predict(X).reshape(-1, 1)
            error = yp - y
            self.b -= 0.01 * np.sum(error) / i
            self.c -= 0.01 * (np.dot(X.T, error).flatten() / i + self.alpha * (self.l1_ratio * np.sign(self.c) + (1 - self.l1_ratio) * 2 * self.c))
        return ElasticNetModelResults(self.c, self.b)

    
    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        return np.dot(X, self.c) + self.b

class ElasticNetModelResults():
    def __init__(self, c, b):
        self.c = c
        self.b = b

    def predict(self, x):
        X = np.array(x, dtype=np.float64)
        return np.dot(x, self.c) + self.b