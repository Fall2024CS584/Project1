import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split 

from elasticnet.models.ElasticNet import ElasticNetModel

def test_predict():
    model = ElasticNetModel(alpha=35, l1_ratio=0.7, learning_rate=0.001, max_iter=5000, tol=1e-5)
    data = []
    column_names = None
    with open("elasticnet/tests/small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        column_names = [k for k in reader.fieldnames if k.startswith('x')]
        for row in reader:
            data.append(row)

    X = np.array([[float(v) for k,v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(v) for datum in data for k,v in datum.items() if k=='y'])
    
    # Standardize the features to improve model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training and testing sets (to evaluate generalization performance)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    results = model.fit(X_train, y_train)
    
    # Predict on both training and testing sets
    preds_train = results.predict(X_train)
    preds_test = results.predict(X_test)
    
    # Calculate Mean Squared Error (MSE) for both train and test sets
    mse_train = np.mean((preds_train - y_train) ** 2)
    mse_test = np.mean((preds_test - y_test) ** 2)
    
    # Calculate R-squared for both train and test sets
    ss_res_train = np.sum((y_train - preds_train) ** 2)
    ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
    r_squared_train = 1 - (ss_res_train / ss_tot_train)
    
    ss_res_test = np.sum((y_test - preds_test) ** 2)
    ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
    r_squared_test = 1 - (ss_res_test / ss_tot_test)

    # Print MSE and R-squared for both training and testing sets
    print(f"Train Mean Squared Error: {mse_train}")
    print(f"Test Mean Squared Error: {mse_test}")
    print(f"Train R-squared: {r_squared_train}")
    print(f"Test R-squared: {r_squared_test}")

    # Print first 5 predictions and actual values for the test set
    print("First 5 test predictions:", preds_test[:5])
    print("First 5 test actual values:", y_test[:5])
    
    # Plot actual vs predicted values for test set against a selected feature (e.g., first feature in X_test)
    feature_index = 0  
    X_test_feature = X_test[:, feature_index]
    feature_name = column_names[feature_index]

    plt.figure(figsize=(10, 6))

    # Plot actual values (y_test) in green
    plt.scatter(X_test_feature, y_test, color='green', label='Actual Values', alpha=0.6)

    # Plot predicted values (y_pred) in blue
    plt.scatter(X_test_feature, preds_test, color='blue', label='Predicted Values', alpha=0.6)

    # Adding labels and title
    plt.xlabel(f'Feature {feature_name}')
    plt.ylabel('Target Value (y)')
    plt.title(f'Feature {feature_name} vs Actual and Predicted Values')

    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    test_predict()