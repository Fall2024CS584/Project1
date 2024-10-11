import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from elasticnet.models.ElasticNet import ElasticNetModel
import matplotlib.pyplot as plt 

def generate_synthetic_data(n_samples=13000, n_features=35, noise=1.0):
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = np.dot(X, true_coef) + noise * np.random.randn(n_samples)
    return X, y

def test_synthetic_data():
    # Generate synthetic data
    X, y = generate_synthetic_data()
        
    # Split the data using sklearn's train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features using StandardScaler from sklearn
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and train the ElasticNet model
    model = ElasticNetModel(alpha=45, l1_ratio=0.4)
    results = model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = results.predict(X_test)
    
    # Evaluate the model using sklearn's metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print the evaluation results
    print("Results for synthetic data:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Plotting y_test and y_pred against the first feature of X_test
    feature_index = 0 
    X_test_feature = X_test[:, feature_index]
    
    plt.figure(figsize=(10, 6))
    
    # Plot actual values (y_test) in green
    plt.scatter(X_test_feature, y_test, color='green', label='Actual Values', alpha=0.6)
    
    # Plot predicted values (y_pred) in blue
    plt.scatter(X_test_feature, y_pred, color='blue', label='Predicted Values', alpha=0.6)
    
    plt.xlabel(f'Feature {feature_index}')
    plt.ylabel('Target Value (y)')
    plt.title(f'Feature {feature_index} vs Actual and Predicted Values')
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_synthetic_data()