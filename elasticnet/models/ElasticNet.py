import numpy as np


class ElasticNetLinearRegression:
    def __init__(self, alpha=0.001, l1_ratio=0.3, max_iter=2000, tol=1e-4, learning_rate=0.01):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.feature_mean = None
        self.feature_std = None

    def _calculate_regularization(self, weights):
        l1_component = self.l1_ratio * np.sum(np.abs(weights))
        l2_component = (1 - self.l1_ratio) * np.sum(weights ** 2)
        return l1_component + l2_component

    def _cost_function(self, X, y, weights, bias):
        num_samples = X.shape[0]
        predictions = X.dot(weights) + bias
        error = y - predictions
        mse_loss = np.sum(error ** 2) / num_samples
        regularization = self.alpha * self._calculate_regularization(weights)
        return mse_loss + regularization

    def _compute_gradients(self, X, y, weights, bias):
        num_samples = X.shape[0]
        predictions = self.predict(X).flatten()
        y = y.flatten()
        residuals = predictions - y

        weight_grad = (2 / num_samples) * X.T.dot(residuals) + self.alpha * (
            self.l1_ratio * np.sign(weights) + (1 - self.l1_ratio) * weights
        )
        bias_grad = (2 / num_samples) * np.sum(residuals)

        return weight_grad, bias_grad


    def fit(self, X, y):

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X = (X - self.mean) / self.std

        n, p = X.shape
        self.weights = np.zeros(p)
        self.bias = 0
        prev_loss = float('inf')

        for i in range(self.max_iter):
            weight_gradient, bias_gradient = self._compute_gradients(
                X, y, self.weights, self.bias)
            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

            loss = self._cost_function(X, y, self.weights, self.bias)

            if np.abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

    def predict(self, X):

        X = (X - self.mean) / self.std
        return X.dot(self.weights) + self.bias

    def r2_score_manual(self, y_true, y_pred):

        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred)

        y_mean = np.mean(y_true)

        tss = np.sum((y_true - y_mean) ** 2)
        rss = np.sum((y_true - y_pred) ** 2)

        r2 = 1 - (rss / tss)

        return r2

    def mae_manual(self, y_true, y_pred):
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        return mae

    def rmse_manual(self, y_true, y_pred):
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred)
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse
