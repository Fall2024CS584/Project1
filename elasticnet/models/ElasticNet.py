import numpy as np

class ElasticNetModel:
    def __init__(self, alpha=0.01, lambda1=0.01, lambda2=0.01, num_iterations=1000, batch_size=32):
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.classes_ = None
        self.models = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))


    def _cost_function(self, X, y, theta):
        m = X.shape[0]
        h = self.sigmoid(np.dot(X, theta))
        h = np.clip(h, 1e-15, 1 - 1e-15)
        cost = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
        reg_cost = self.lambda1 * np.sum(np.abs(theta)) + self.lambda2 * np.sum(np.square(theta))
        return cost + reg_cost

    def _gradient(self, X, y, theta):
        m = X.shape[0]
        h = self.sigmoid(np.dot(X, theta))
        gradient = (1/m) * np.dot(X.T, (h - y))
        gradient += self.lambda2 * theta
        gradient += self.lambda1 * np.sign(theta)
        return gradient

    def _fit_binary(self, X, y):
        m, n = X.shape
        theta = np.zeros((n, 1))
        for iteration in range(self.num_iterations):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                gradient = self._gradient(X_batch, y_batch, theta)
                theta -= self.alpha * gradient
        return theta

    def fit(self, X, y):
        X = X.astype(float)
        y = y.astype(float).reshape(-1, 1)
        
        self.classes_ = np.unique(y)
        self.models = []
        for c in self.classes_:
            y_binary = (y == c).astype(int)
            theta = self._fit_binary(X, y_binary)
            self.models.append(theta)
        return ElasticNetModelResults(self.models, self.classes_)

class ElasticNetModelResults:
    def __init__(self, models, classes):
        self.models = models
        self.classes = classes

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        X = X.astype(float)
        
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, theta in enumerate(self.models):
            predictions[:, i] = self.sigmoid(np.dot(X, theta)).ravel()
        
        # Convert probabilities to predicted classes
        if len(self.models) == 1:  # Binary classification case
            return (predictions >= 0.5).astype(float)
        
        # Multiclass case, returning class labels based on max probability
        return self.classes[np.argmax(predictions, axis=1)]                                                                 