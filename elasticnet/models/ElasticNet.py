# -*- coding: utf-8 -*-
"""ML Project-1.ipynb
Group Members:
A20584318 - ANSH KAUSHIK
A20593046 - ARUNESHWARAN SIVAKUMAR
A20588339 - HARISH NAMASIVAYAM MUTHUSWAMY
A20579993 - SHARANYA MISHRA
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.sparse import csr_matrix

# Function to split data into training and testing sets (75% training, 25% testing)
def train_test_split(X, y, test_size=0.25, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    split_index = int((1 - test_size) * len(indices))
    train_indices, test_indices = indices[:split_index], indices[split_index:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

# ElasticNet Regression (combination of L1 and L2 regularization)
class ElasticNetRegression:
    def __init__(self, alpha, l1_ratio, iterations, learning_rate):
        self.alpha = alpha  # Overall regularization strength
        self.l1_ratio = l1_ratio  # Balance between L1 and L2 regularization
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Add a bias term to X (intercept)
        X = np.c_[np.ones(X.shape[0]), X]
        n_samples, n_features = X.shape

        # Initialize weights (coefficients)
        self.coef_ = np.zeros(n_features)

        # Gradient Descent
        for i in range(self.iterations):
            y_pred = X.dot(self.coef_)
            residuals = y_pred - y

            # Calculate gradient
            gradient = X.T.dot(residuals) / n_samples
            l1_penalty = self.l1_ratio * np.sign(self.coef_)
            l2_penalty = (1 - self.l1_ratio) * self.coef_

            # Update coefficients with combined L1 (lasso) and L2 (ridge) penalties
            self.coef_ -= self.learning_rate * (gradient + self.alpha * (l1_penalty + l2_penalty))

            # Check if the coefficients are diverging by printing every 1000 iterations
            if i % 1000 == 0:
                print(f"Iteration {i}: Coefficients: {self.coef_}")

        # Store the intercept separately for easier prediction
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def predict(self, X):
        # Add bias term to X (for intercept)
        X = np.c_[np.ones(X.shape[0]), X]
        return X.dot(np.r_[self.intercept_, self.coef_])

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        residual_variance = np.sum((y_true - y_pred) ** 2)
        return 1 - (residual_variance / total_variance)