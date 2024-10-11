import unittest
import numpy as np

from regression_models import normalize_features, elastic_net_regression, ridge_regression, lasso_regression, polynomial_regression

class TestRegressionModels(unittest.TestCase):

    def setUp(self):
        """Set up sample data for testing"""
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        self.y = np.dot(self.X, np.array([3, 1.5, -2])) + np.random.randn(100)
        self.X_normalized, _, _ = normalize_features(self.X)
        self.X_train = self.X_normalized[:80]
        self.y_train = self.y[:80]
        self.X_test = self.X_normalized[80:]
        self.y_test = self.y[80:]

    def test_elastic_net_regression(self):
        """Test ElasticNet regression with fixed alpha and l1_ratio"""
        coeff = elastic_net_regression(self.X_train, self.y_train, alpha=0.01, l1_ratio=0.9)
        y_pred = self.X_test @ coeff
        mse = np.mean((self.y_test - y_pred) ** 2)
        self.assertLess(mse, 1.0, "ElasticNet regression MSE is too high")

    def test_ridge_regression(self):
        """Test Ridge regression with fixed alpha"""
        coeff = ridge_regression(self.X_train, self.y_train, alpha=1.0)
        y_pred = self.X_test @ coeff
        mse = np.mean((self.y_test - y_pred) ** 2)
        self.assertLess(mse, 1.0, "Ridge regression MSE is too high")

    def test_lasso_regression(self):
        """Test Lasso regression with fixed alpha"""
        coeff = lasso_regression(self.X_train, self.y_train, alpha=1.0)
        y_pred = self.X_test @ coeff
        mse = np.mean((self.y_test - y_pred) ** 2)
        self.assertLess(mse, 1.0, "Lasso regression MSE is too high")

    def test_polynomial_regression(self):
        """Test Polynomial regression for a fixed degree"""
        coeff, X_train_poly = polynomial_regression(self.X_train, self.y_train, degree=2)
        X_test_poly = np.hstack([self.X_test ** i for i in range(1, 3)])
        y_pred = X_test_poly @ coeff
        mse = np.mean((self.y_test - y_pred) ** 2)
        self.assertLess(mse, 1.0, "Polynomial regression MSE is too high")

if __name__ == "__main__":
    unittest.main()
