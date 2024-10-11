import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ElasticNetModel code
class ElasticNetModel:
    def __init__(self, alpha=1.0, l1_ratio=0.5, convergence_threshold=1e-4, max_iterations=1000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.coefficients_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Adding a bias term to the features
        X = np.c_[np.ones(X.shape[0]), X]  # Bias term (intercept)
        num_samples, num_features = X.shape
        
        # Initializing coefficients to zero
        self.coefficients_ = np.zeros(num_features)
        
        for _ in range(self.max_iterations):
            predictions = X @ self.coefficients_
            residuals = y - predictions
            
            # Updating coefficients using the ElasticNet regularization
            for index in range(num_features):
                if index == 0:  # Intercept term
                    gradient = -2 * np.sum(residuals) / num_samples
                    self.coefficients_[index] -= self.alpha * gradient
                else:
                    gradient = -2 * (X[:, index] @ residuals) / num_samples
                    l1_penalty = self.l1_ratio * self.alpha * np.sign(self.coefficients_[index])
                    l2_penalty = (1 - self.l1_ratio) * self.alpha * self.coefficients_[index]
                    self.coefficients_[index] -= self.alpha * (gradient + l1_penalty + l2_penalty)
            
            # Checking for convergence
            if np.sum(np.abs(gradient)) < self.convergence_threshold:
                break

        self.intercept_ = self.coefficients_[0]
        self.coefficients_ = self.coefficients_[1:]

    def predict(self, X):
        return self.intercept_ + X @ self.coefficients_

# Function to compute Root Mean Squared Error (RMSE)
def compute_rmse(actual_values, predicted_values):
    return np.sqrt(np.mean((actual_values - predicted_values) ** 2))

# Main execution of the code
if __name__ == "__main__":
    # Reading data from CSV file
    dataset = pd.read_csv('output.csv')
    features = dataset[['x_0', 'x_1']].values
    target = dataset['y'].values

    # Splitting the dataset into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Normalizing the features
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)

    # Initializing and train the ElasticNet regression model
    elastic_net = ElasticNetModel(alpha=0.01, l1_ratio=0.7)  # Adjusted learning rate
    elastic_net.fit(X_train, y_train)

    # Predicting the target values for the test set and evaluate the model's performance
    y_pred = elastic_net.predict(X_test)
    test_rmse = compute_rmse(y_test, y_pred)
    print("Test RMSE:", test_rmse)
