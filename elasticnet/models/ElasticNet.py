import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ElasticNet():
    """
    This class implements the Elastic Net regression model.
    It merges L1 (Lasso) and L2 (Ridge) regularization methods.
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        """
        Initializes the model with parameters:
        alpha: Controls the strength of regularization.
        l1_ratio: Regulates the relative contribution of L1 and L2 regularization techniques.
        max_iter: Sets the maximum number of iterations for training.
        tol: The tolerance for stopping the iteration.
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None # Model's weight values
        self.intercept_ = None  # Intercept (bias) of the model


    def fit(self, X, y):
        """
        Trains the Elastic Net model using the data (X) and target values (y).
        X: The input data (features).
        y: The target values.
        """
        # Check if the dimensions of X and y are valid
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X : 2DARRAY and y : 1DARRAY.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X must match y.")
        
        # Check for zero variance in features
        if np.any(np.var(X, axis=0) == 0):
            raise ValueError("Feature variance cannot be 0.")

        #Append a column of ones to X to account for the intercept (bias term)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        n_samples, n_features = X.shape  # Get the shape of X
        self.coef_ = np.zeros(n_features)  # Initialize the coefficients with zeros

        # Iteratively update the coefficients
        for iteration in range(self.max_iter):
            coef_prev = self.coef_.copy()  # Save the previous coefficients

            for j in range(n_features):
                # Calculate the residual (error) without feature j
                residual = y - np.dot(X, self.coef_) + X[:, j] * self.coef_[j]
                rho = np.dot(X[:, j], residual)  # Calculate rho, a key term

                if j == 0:
                    # Update intercept (no regularization)
                    self.coef_[j] = rho / np.sum(X[:, j]**2)
                else:
                    z = np.sum(X[:, j]**2)  # Sum of squares of feature j
                    l1_penalty = self.alpha * self.l1_ratio  # L1 regularization term
                    l2_penalty = self.alpha * (1 - self.l1_ratio)  # L2 regularization term
                    # Update the coefficient with L1 and L2 penalties
                    self.coef_[j] = np.sign(rho) * max(0, abs(rho) - l1_penalty) / (z + l2_penalty)

            ## Halt the process if coefficient change falls below the tolerance level.
            if np.linalg.norm(self.coef_ - coef_prev) < self.tol:
                break

        # Separate intercept from the rest of the coefficients
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

        return self  # Allows method chaining by returning the instance
    
    def predict(self, X):
        """
        Predicts the target values for the input data X using the trained model.
        
        X: The input data (features).
        
        Returns the predicted values.
        """
        return np.dot(X, self.coef_) + self.intercept_

    def evaluate(self, X, y_true):
        """
        Evaluates the model's performance using MSE, MAE, and R-squared.
        
        X: Input data (features).
        y_true: The actual target values.

        Returns the MSE, MAE, and R-squared values.
        """
        y_pred = self.predict(X)  # Get the predictions

        # Calculate Mean Squared Error (MSE)
        mse = np.average((y_true - y_pred) ** 2)

        # Calculate Mean Absolute Error (MAE)
        mae = np.mean(np.abs(y_true - y_pred))

        # Calculate R-squared (R²)
        sst_value = np.sum((y_true - np.mean(y_true)) ** 2)
        ssr_value = np.sum((y_true - y_pred) ** 2)
        r_squared_value = 1 - (ssr_value / sst_value)

        return mse, mae, r_squared_value

    def plot_predictions(self, y_true, y_pred):
        """Plot(residuals) predicted vs actual values."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.7, color='blue', label='Predicted vs Original')
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal Prediction')
        plt.xlabel('Original Values')
        plt.ylabel('Predicted Values')
        plt.title('Original vs. Predicted Values')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_residuals(self, y_true, y_pred):
        """Graph(residuals) predicted vs actual outcomes."""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.7, color='green')
        plt.axhline(0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs. Predicted Values')
        plt.grid()
        plt.show()

# Generating LDA(linear Random Data)
def linear_data_generator(m, b, rnge, N, scale, seed):
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(low=rnge[0], high=rnge[1], size=(N, m.shape[0]))
    ys = np.dot(sample, np.reshape(m, (-1, 1))) + b
    noise = rng.normal(loc=0., scale=scale, size=ys.shape)
    return sample, (ys + noise).flatten()
    
def run_elastic_net(X, y, alpha=0.1, l1_ratio=0.2):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000, tol=1e-4)
    model.fit(X, y)
    predictions = model.predict(X)
    mse, mae, r_squared = model.evaluate(X, y)
      
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r_squared:.4f}")
        
    model.plot_predictions(y, predictions)
    model.plot_residuals(y, predictions)
    return model

def main():
    while True:
        print("\nSelect an option:")
        print("1. Generate synthetic data")
        print("2. Load data from CSV")
        print("3. Exit")
        choice = input("Enter choice: ")

        if choice == '1':
            try:
                m = np.array([1, 2])
                b = 5
                N = int(input("Number of samples: "))
                scale = float(input("Noise scale: "))
                seed = int(input("Random seed: "))

                X, y = linear_data_generator(m, b, (0, 10), N, scale, seed)
                run_elastic_net(X, y)
            except ValueError as e:
                print(f"Input error: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

        elif choice == '2':
            try:
                CSV_FILEDATA = input("Please enter CSV only file path: ")
                df = pd.read_csv(CSV_FILEDATA)
                if 'y' not in df.columns:
                    raise ValueError("'y' column not found in CSV.")

                X = df.drop('y', axis=1).values
                y = df['y'].values
                run_elastic_net(X, y)
            except FileNotFoundError:
                print("File not found.")
            except ValueError as e:
                print(f"Input error: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

        elif choice == '3':
            print("Exiting.")
            break

        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()