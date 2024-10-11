import numpy as np
from numba import jit
from typing import Tuple

class ElasticNetModel:
    def __init__(
            self,
            learning_rate: float = 0.01,
            iterations: int = 1000,
            l1_ratio: float = 0.5,
            alpha: float = 1.0) -> None:
        
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.weights = np.empty(0)
        self.bias = 0.0

    def fit(
            self,
            features: np.ndarray,
            target: np.ndarray) -> None:
        
        num_samples, num_features = features.shape
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        self.weights, self.bias = self._optimize(features, target, self.weights, self.bias, self.learning_rate, self.iterations, self.alpha, self.l1_ratio, num_samples)

    @staticmethod
    @jit(nopython=True, nogil=True)
    def _optimize(
        features: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
        bias: float,
        learning_rate: float,
        iterations: int,
        alpha: float,
        l1_ratio: float,
        num_samples: int) -> Tuple[np.ndarray, float]:
        
        for _ in range(iterations):
            predictions = np.dot(features, weights) + bias
            errors = predictions - target

            l2_gradient = 2 * weights
            l1_gradient = np.sign(weights)

            weights -= learning_rate * ((1 / num_samples) * np.dot(features.T, errors) + alpha * ((1 - l1_ratio) * l2_gradient + l1_ratio * l1_gradient))
            bias -= learning_rate * (1 / num_samples) * np.sum(errors)

        return weights, bias

    def predict(self, features: np.ndarray) -> np.ndarray:
        predictions = np.dot(features, self.weights) + self.bias
        return predictions
