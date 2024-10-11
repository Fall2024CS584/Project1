import pandas as pd
import numpy as np
import csv
import os
from models import LassoModel, RidgeModel, ElasticNetModel, RegularizedRegression
from models.decision_tree import DecisionTreeRegressor


class KNNModel:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))  # Euclidean distance
            k_indices = distances.argsort()[:self.k]  # Get indices of k closest points
            k_nearest_labels = self.y_train[k_indices]
            # Majority vote (regression case - take the mean)
            prediction = np.mean(k_nearest_labels)
            predictions.append(prediction)
        return np.array(predictions)

def run_models():
   for i in range(3):
        if i == 1:
            print("\n ")
            print("----------------------------WHITE WINE----------------------------")
            print("\n ")
            data = pd.read_csv('winequality-white.csv', sep=';')
            X = data.drop(columns='quality').values  
            y = data['quality'].values
        elif i == 0:
            print("\n ")
            print("----------------------------DIABETES----------------------------")
            print("\n ")
            data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', header=None)
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
        elif i == 2:
            print("\n ")
            print("----------------------------GENERATED DATA----------------------------")
            print("\n ")
            data = pd.read_csv('Data.csv')
            X = data.drop(columns='y').values  
            y = data['y'].values

        data = data.dropna()

        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_scaled = (X - X_mean) / X_std
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]  # Add bias term

        models = [
            ("Linear Regression", RegularizedRegression(), {}),
            ("Ridge Regression", RidgeModel(lambda_l2=0.01, alpha=0.001, num_iterations=1000), {}),
            ("Lasso Regression", LassoModel(lambda_l1=0.01, alpha=0.001, num_iterations=1000), {}),
            ("Elastic Net Regression", ElasticNetModel(lambda_l1=0.01, lambda_l2=0.01, alpha=0.001, num_iterations=1000), {}),
            ("KNN Regression", KNNModel(k=5), {})
        ]

        for model_name, model, _ in models:
            model.fit(X_b, y)
            y_pred = model.predict(X_b)
            mse = (np.square(y - y_pred)).mean(axis=None)
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            # Print and store the results for each model
            print(f"{model_name} MSE: {mse:.4f}, R²: {r2:.4f}")


def calculate_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

def test_predict():
    print("\n ")
    print("----------------------------Test Data----------------------------")
    print("\n ")

    models = {
        "Lasso": LassoModel(),
        "Ridge": RidgeModel(),
        "KNN": KNNModel(),
        "ElasticNet": ElasticNetModel()
    }

    data = []

    current_dir = os.getcwd()

    file_path = os.path.join(current_dir, "tests", "test.csv")
    try:
        with open(file_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    
    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])
    
    for name, model in models.items():
        print("\n ")
        print(f"Testing {name}")
        print("\n ")
        model.fit(X, y)
        preds = model.predict(X)
        print(f"Predictions for {name}: {preds}")
        
        # Calculating Mean Squared Error (MSE)
        mse = np.mean((preds - y) ** 2)
        print(f"Mean Squared Error for {name}: {mse}")
        
        # Calculate R² score
        r2 = calculate_r2(y, preds)
        print(f"R² for {name}: {r2:.4f}")
        

        assert mse < 22.0, f"High MSE for {name}: MSE = {mse}"


if __name__ == "__main__":
    run_models()  # Run model training
    test_predict()  # Run prediction test
