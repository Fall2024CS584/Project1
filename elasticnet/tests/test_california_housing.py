import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from elasticnet.models.ElasticNet import ElasticNetModel

class ElasticNetTest:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
    
    def load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        
        # Drop the ocean_proximity column as it's categorical
        df = df.drop('ocean_proximity', axis=1)
        
        # Handle missing values if any
        df = df.fillna(df.mean())
        
        self.feature_names = df.drop('median_house_value', axis=1).columns.tolist()
        
        # Separate features and target
        X = df.drop('median_house_value', axis=1)
        y = df['median_house_value']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Convert to numpy arrays and ensure correct shapes
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        
        # Reshape y values to match model expectations (flatten to 1D array)
        self.y_train = np.array(self.y_train).flatten()
        self.y_test = np.array(self.y_test).flatten()
        
        return self
    
    def train_and_evaluate(self):
        # Initialize and train the model
        self.model = ElasticNetModel(
            alpha=0.1,
            l1_ratio=0.5,
            learning_rate=0.01,
            max_iter=1000,
            tol=1e-4
        )
        
        # Fit the model
        results = self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        train_predictions = results.predict(self.X_train)
        test_predictions = results.predict(self.X_test)
        
        # Reshape predictions if needed for metric calculations
        train_predictions = np.array(train_predictions).flatten()
        test_predictions = np.array(test_predictions).flatten()
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(self.y_train, train_predictions),
            'test_mse': mean_squared_error(self.y_test, test_predictions),
            'train_r2': r2_score(self.y_train, train_predictions),
            'test_r2': r2_score(self.y_test, test_predictions)
        }
        
        return metrics
    
    def plot_test_vs_pred(self, X_test, y_test, y_pred):
        feature_index = 0
        X_test_feature = X_test[:, feature_index]
        
        feature_name = self.feature_names[feature_index]

        # Plot y_test and y_pred against the selected feature from X_test
        plt.figure(figsize=(10, 6))

        # Plot actual values (y_test) in green
        plt.scatter(X_test_feature, y_test, color='green', label='Actual Values', alpha=0.6)

        # Plot predicted values (y_pred) in blue
        plt.scatter(X_test_feature, y_pred, color='blue', label='Predicted Values', alpha=0.6)

        plt.xlabel(f'Feature {feature_name}')
        plt.ylabel('Median House Value')
        plt.title(f'Feature {feature_name} vs Actual and Predicted Median House Value')

        plt.legend()
        plt.show()

def run_test():
    # Initialize test
    test = ElasticNetTest('elasticnet/tests/california_housing.csv')
    
    # Preprocess data
    test.load_and_preprocess_data()
    
    # Train and evaluate
    metrics = test.train_and_evaluate()
    
    # Print results
    print("\nModel Evaluation Metrics:")
    print(f"Training MSE: {metrics['train_mse']:.2f}")
    print(f"Test MSE: {metrics['test_mse']:.2f}")
    print(f"Training R-squared: {metrics['train_r2']:.4f}")
    print(f"Test R-squared: {metrics['test_r2']:.4f}")
    
    # Generate test predictions again for plotting
    test_predictions = test.model.fit(test.X_train, test.y_train).predict(test.X_test)
    
    # Plot the results
    test.plot_test_vs_pred(test.X_test, test.y_test, test_predictions)

if __name__ == "__main__":
    run_test()