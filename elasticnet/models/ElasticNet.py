import numpy as np

class ElasticNetModel():
    def __init__(self, alpha=0.001, l1_ratio=0.5, max_iter=1000, tol=1e-5):
        self.alpha = alpha 
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for i in range(self.max_iter):
            y_pred = self._predict(X)

            gradient_weight = -(1/n_samples) * X.T.dot(y - y_pred) + self.alpha * ((1 - self.l1_ratio) * 2 * self.weight + self.l1_ratio* np.sign(self.weight))
            gradient_bias= -(1/n_samples) * np.sum(y - y_pred)

            weight_old = self.weight.copy()
            self.weight -= self.alpha * gradient_weight
            self.bias -= self.alpha * gradient_bias

            if np.sum(np.abs(self.weight - weight_old))<self.tol:
                break

    def predict(self, X):
        return self._predict(X)

    def _predict(self, X):
        return X.dot(self.weight) + self.bias