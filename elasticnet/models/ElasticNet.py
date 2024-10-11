import csv
import numpy as np

class ElasticNetModel():
    def __init__(self, alpha=1.0, l1_ratio=0.3, lrate=0.02, n_iters=200):
        self.alpha = alpha  
        self.l1_ratio = l1_ratio  
        self.lrate = lrate  
        self.n_iters = n_iters  
        self.weights = None  

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)  
        
        for _ in range(self.n_iters):
            pred = X.dot(self.weights)

            errors = pred - y
            gradient = (X.T.dot(errors) / len(y)) + \
                    (self.alpha * (1 - self.l1_ratio) * self.weights) + \
                    (self.alpha * self.l1_ratio * np.sign(self.weights))
            
            self.weights -= self.lrate * gradient

        return ElasticNetModelResults(self.weights)


class ElasticNetModelResults():
    def __init__(self, coef):
        self.weights = coef

    def predict(self, X):
        return X @ self.weights
