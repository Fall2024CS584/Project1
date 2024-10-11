import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from elasticnet.models.ElasticNet import ElasticNetModel

class ElasticNetStockTest:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = None
        self.scaler = StandardScaler()
    
    def create_features(self, df):
        # Create technical indicators
        df['Returns'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Price_Range'] = df['High'] - df['Low']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Create target variable (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        return df
    
    def load_and_preprocess_data(self):
        # Load the data
        df = pd.read_csv(self.data_path, parse_dates=['Date'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Create features
        df = self.create_features(df)
        
        # Select features for model
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'MA5', 'MA20', 'Volatility', 'Price_Range', 'Volume_Change'
        ]
        
        X = df[self.feature_columns]
        y = df['Target']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the data chronologically
        train_size = int(len(df) * 0.85)
        self.X_train = X_scaled[:train_size]
        self.X_test = X_scaled[train_size:]
        self.y_train = np.array(y[:train_size]).flatten()
        self.y_test = np.array(y[train_size:]).flatten()
        
        return self
    
    def train_and_evaluate(self):
        # Initialize and train the model
        self.model = ElasticNetModel(
            alpha=40,  
            l1_ratio=0.6,
            learning_rate=0.001,  
            max_iter=2000, 
            tol=1e-5
        )
        
        results = self.model.fit(self.X_train, self.y_train)
        
        train_predictions = results.predict(self.X_train)
        test_predictions = results.predict(self.X_test)
        
        # Reshape predictions
        train_predictions = np.array(train_predictions).flatten()
        test_predictions = np.array(test_predictions).flatten()
        
        metrics = {
            'train_mse': mean_squared_error(self.y_train, train_predictions),
            'test_mse': mean_squared_error(self.y_test, test_predictions),
            'train_r2': r2_score(self.y_train, train_predictions),
            'test_r2': r2_score(self.y_test, test_predictions)
        }
        
        return metrics, results, test_predictions 
    
    def plot_test_vs_pred(self, X_test, y_test, y_pred):
        # Select a feature for plotting, e.g., 'Close'
        feature_index = self.feature_columns.index('Close')  
        X_test_feature = X_test[:, feature_index]

        plt.figure(figsize=(10, 6))

        # Plot actual values (y_test) in green
        plt.scatter(X_test_feature, y_test, color='green', label='Actual Values', alpha=0.6)

        # Plot predicted values (y_pred) in blue
        plt.scatter(X_test_feature, y_pred, color='blue', label='Predicted Values', alpha=0.6)

        plt.xlabel('Close Price (Test Set)')
        plt.ylabel('Target Value (y)')
        plt.title('Close Price vs Actual and Predicted Values')

        plt.legend()
        plt.show()

def run_test():
    # Initialize test
    test = ElasticNetStockTest('elasticnet/tests/netflix.csv')
    
    # Preprocess data
    test.load_and_preprocess_data()
    
    # Train and evaluate
    metrics, results, test_predictions  = test.train_and_evaluate()
    
    print("\nModel Evaluation Metrics:")
    print(f"Training MSE: {metrics['train_mse']:.2f}")
    print(f"Test MSE: {metrics['test_mse']:.2f}")
    print(f"Training R square: {metrics['train_r2']:.4f}")
    print(f"Test R square: {metrics['test_r2']:.4f}")
    
    test.plot_test_vs_pred(test.X_test, test.y_test, test_predictions)
    

if __name__ == "__main__":
    run_test()