import numpy as np
import pandas as pd
import pickle
import unittest
import matplotlib.pyplot as plt
import time

class ElasticNetModel():
    def __init__(self, alpha=1.0, l1_ratio=0.5, learning_rate=0.001, num_iterations=1000, tolerance=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.convergence_time = None

    def _elasticnet_penalty(self, weights):
        l1_penalty = self.l1_ratio * np.sum(np.abs(weights))
        l2_penalty = (1 - self.l1_ratio) * np.sum(weights ** 2)
        return self.alpha * (l1_penalty + l2_penalty)

    def _compute_cost(self, X, y, weights, bias):
        m = X.shape[0]
        predictions = X.dot(weights) + bias
        residuals = predictions - y
        mse = (1 / (2 * m)) * np.sum(residuals ** 2)
        regularization = self._elasticnet_penalty(weights)
        return mse + regularization

    def _compute_gradients(self, X, y, weights, bias):
        m = X.shape[0]
        predictions = X.dot(weights) + bias
        residuals = predictions - y

        dw = (1 / m) * X.T.dot(residuals) + self.alpha * (self.l1_ratio * np.sign(weights) + (1 - self.l1_ratio) * weights)
        db = (1 / m) * np.sum(residuals)
        return dw, db

    def fit(self, X, y):
        # Check for missing values and raise an error if found
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Input data contains NaN values. Please handle missing data before training.")

        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Start timing the convergence
        start_time = time.time()

        for i in range(self.num_iterations):
            dw, db = self._compute_gradients(X, y, self.weights, self.bias)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            cost = self._compute_cost(X, y, self.weights, self.bias)
            
            # Early stopping if cost change is less than the tolerance
            if i > 0 and abs(self.cost_history[-1] - cost) < self.tolerance:
                print(f"Early stopping at iteration {i} due to small cost change.")
                break

            self.cost_history.append(cost)

        # End timing
        self.convergence_time = time.time() - start_time
        return ElasticNetModelResults(self.weights, self.bias)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'weights': self.weights, 'bias': self.bias}, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
            self.weights = model_data['weights']
            self.bias = model_data['bias']

class ElasticNetModelResults():
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, X):
        return X.dot(self.weights) + self.bias

# Custom scaling function
def custom_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

# Custom train-test split function
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_index = int((1 - test_size) * len(y))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Function to preprocess data, ensuring only numerical data is used
def preprocess_data(data):
    data = data.select_dtypes(include=[np.number])
    data = data.dropna()
    return data

def run_test():
    # Load the dataset (replace with your actual file path)
    data = pd.read_csv('/content/small_test.csv')
    data = preprocess_data(data)

    # Assuming the last column is the target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X, mean, std = custom_scaler(X)
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

    model = ElasticNetModel(alpha=0.5, l1_ratio=0.5, learning_rate=0.01, num_iterations=1000, tolerance=1e-4)

    trained_model = model.fit(X_train, y_train)

    predictions = trained_model.predict(X_test)

    plt.plot(range(len(model.cost_history)), model.cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function Convergence')
    plt.show()

    print("Predictions:", predictions[:5])
    print("Actual values:", y_test[:5])
    print(f"Model converged in {model.convergence_time:.4f} seconds.")

    model.save_model('trained_model.pkl')
    print("Model saved!")

    loaded_model = ElasticNetModel(alpha=0.5, l1_ratio=0.5, learning_rate=0.01, num_iterations=1000)
    loaded_model.load_model('trained_model.pkl')
    print("Model loaded!")

# Unit test class
class TestElasticNetModel(unittest.TestCase):

    def plot_cost_convergence(self, model, title):
        # Function to plot cost convergence
        plt.plot(range(len(model.cost_history)), model.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title(title)
        plt.show()

    def test_training_on_synthetic_data(self):
        np.random.seed(42)
        X_synthetic = np.random.rand(100, 3)
        y_synthetic = X_synthetic.dot(np.array([2, -1, 3])) + np.random.randn(100) * 0.5

        model = ElasticNetModel(alpha=0.5, l1_ratio=0.5, learning_rate=0.01, num_iterations=1000)
        trained_model = model.fit(X_synthetic, y_synthetic)
        predictions = trained_model.predict(X_synthetic)

        self.assertEqual(predictions.shape, y_synthetic.shape)
        self.assertLess(model.cost_history[-1], model.cost_history[0])

        # Plot cost convergence for this test case
        self.plot_cost_convergence(model, "Cost Convergence: Synthetic Data")

    def test_save_and_load_model(self):
        np.random.seed(42)
        X_synthetic = np.random.rand(100, 3)
        y_synthetic = X_synthetic.dot(np.array([2, -1, 3])) + np.random.randn(100) * 0.5

        model = ElasticNetModel(alpha=0.5, l1_ratio=0.5, learning_rate=0.01, num_iterations=1000)
        model.fit(X_synthetic, y_synthetic)

        model.save_model('test_model.pkl')

        loaded_model = ElasticNetModel(alpha=0.5, l1_ratio=0.5, learning_rate=0.01, num_iterations=1000)
        loaded_model.load_model('test_model.pkl')

        np.testing.assert_array_equal(model.weights, loaded_model.weights)
        self.assertEqual(model.bias, loaded_model.bias)

if __name__ == "__main__":
    run_test()
    unittest.main(argv=[''], exit=False)
