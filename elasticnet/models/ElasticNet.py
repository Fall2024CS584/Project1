import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class LinearRegression:
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    # Function for model training
    def fit(self, X, Y):
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape
        # weight initialization
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # gradient descent learning
        for i in range(self.iterations):
            self.update_theta()

        return self

    # Helper function to update weights in gradient descent
    def update_theta(self):
        Y_pred = self.predict(self.X)
        # calculate gradients
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m
        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return self

    # Hypothetical function  h( x )
    def predict(self, X):
        return X.dot(self.W) + self.b


# Elastic Net Regression
class MyElasticNetImplementation():
    def __init__(self, learning_rate, iterations, l1_penalty, l2_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

    # Function for model training
    def fit(self, X, Y):
        # no_of_training_examples, no_of_features
        self.n_samples, self.n_features = X.shape
        # weight initialization
        self.W = np.zeros(self.n_features)
        self.b = 0
        self.X = np.array(X)
        self.Y = np.array(Y)

        # gradient descent learning
        for i in range(self.iterations):
            self.update_theta()

        return self

    # Helper function to update weights in gradient descent
    def update_theta(self):
        Y_pred = self.predict(self.X)
        error = self.Y - Y_pred
        # calculate gradients
        dW = np.zeros(self.n_features)

        for j in range(self.n_features):
            dW[j] = (-(2 * (self.X[:, j]).dot(self.Y - Y_pred)) +
                     np.sign(self.W[j]) * self.l1_penalty + 2 * self.l2_penalty * self.W[j]) / self.n_samples

        db = - 2 * np.sum(self.Y - Y_pred) / self.n_features

        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return self

    # Hypothetical function h( x )
    def predict(self, X):
        return X.dot(self.W) + self.b

    def find_optimal_penalties(X_train, Y_train, X_test, Y_test):
        best_l1 = 0
        best_l2 = 0
        best_mse = float('inf')  # Set to infinity initially

        # Define the ranges of l1 and l2 penalties to search
        l1_penalty_range = np.concatenate([
            np.logspace(-4, -2, 10),  # Very small values: 0.0001 to 0.01
            np.logspace(-2, 0, 20),   # Small to medium values: 0.01 to 1
            np.logspace(0, 2, 10)     # Large values: 1 to 100
        ])

        l2_penalty_range = np.concatenate([
            np.logspace(-4, -2, 10),  # Very small values: 0.0001 to 0.01
            np.logspace(-2, 0, 20),   # Small to medium values: 0.01 to 1
            np.logspace(0, 2, 10)     # Large values: 1 to 100
        ])

        print(l1_penalty_range.shape)
        print(l2_penalty_range.shape)

        # Remove duplicate values and sort
        l1_penalty_range = np.unique(l1_penalty_range)
        l2_penalty_range = np.unique(l2_penalty_range)

        for l1_penalty in l1_penalty_range:
            for l2_penalty in l2_penalty_range:
                # Train the Elastic Net model with current penalties
                model = MyElasticNetImplementation(learning_rate=0.01, iterations=1000, l1_penalty=l1_penalty, l2_penalty=l2_penalty)
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)

                # Calculate Mean Squared Error
                mse = mean_squared_error(Y_test, Y_pred)

                # If this is the best MSE so far, update best penalties
                if mse < best_mse:
                    best_mse = mse
                    best_l1 = l1_penalty
                    best_l2 = l2_penalty

        return best_l1, best_l2, best_mse


def main():
    # Importing dataset
    df = pd.read_csv("salary_data.csv")
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, 1].values

    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

    # Find optimal penalties for Elastic Net
    best_l1, best_l2, best_mse = MyElasticNetImplementation.find_optimal_penalties(X_train, Y_train, X_test, Y_test)
    print(f"Optimal L1 penalty: {best_l1}")
    print(f"Optimal L2 penalty: {best_l2}")
    print(f"Best Mean Squared Error for Elastic Net: {best_mse}")

    # Linear Regression Model training
    lin_reg_model = LinearRegression(iterations=1000, learning_rate=0.01)
    lin_reg_model.fit(X_train, Y_train)

    # Elastic Net Regression Model training
    elastic_reg_model = MyElasticNetImplementation(learning_rate=0.01, iterations=1000, l1_penalty=best_l1, l2_penalty=best_l2)
    elastic_reg_model.fit(X_train, Y_train)

    # Prediction on test set
    lin_Y_pred = lin_reg_model.predict(X_test)
    elastic_Y_pred = elastic_reg_model.predict(X_test)

    # Output comparison
    print("Linear Regression Predictions:", np.round(lin_Y_pred[:3], 2))
    print("Elastic Net Regression Predictions:", np.round(elastic_Y_pred[:3], 2))
    print("Real values:", Y_test[:3])
    print("Linear Regression Trained W:", round(lin_reg_model.W[0], 2))
    print("Elastic Net Regression Trained W:", round(elastic_reg_model.W[0], 2))
    print("Linear Regression Trained b:", round(lin_reg_model.b, 2))
    print("Elastic Net Regression Trained b:", round(elastic_reg_model.b, 2))

    # Visualization
    plt.figure(figsize=(10, 5))

    # Linear Regression plot
    plt.subplot(1, 2, 1)
    plt.scatter(X_test, Y_test, color='blue')
    plt.plot(X_test, lin_Y_pred, color='orange')
    plt.title('Linear Regression: Salary vs Experience')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')

    # Elastic Net Regression plot
    plt.subplot(1, 2, 2)
    plt.scatter(X_test, Y_test, color='blue')
    plt.plot(X_test, elastic_Y_pred, color='green')
    plt.title('Elastic Net Regression: Salary vs Experience')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
