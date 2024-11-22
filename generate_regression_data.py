import numpy as np
import csv
import argparse
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt


def load_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        X, y = [], []
        for row in reader:
            X.append([float(num) for num in row[:-1]])
            y.append(float(row[-1]))
        return np.array(X), np.array(y)

def test_model_with_generated_data(filename):
    X, y = load_data(filename)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = ElasticNetModel(lambdas=1.0, l1_ratio=0.5, iterations=1000, learning_rate=0.01)
    results = model.fit(X_scaled, y)
    
    predictions = results.predict(X_scaled)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions, alpha=0.5)
    plt.title('Comparison of Actual and Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # Diagonal line for reference
    plt.grid(True)
    plt.show()
    
    print("Predictions:", predictions)
    print("Actuals:", y)
    return predictions, y
    
predictions, actuals = test_model_with_generated_data('data.csv')


def generate_data(N, m, b, scale, rnge, seed, output_file):
    def linear_data_generator(m, b, rnge, N, scale, seed):
        rng = np.random.default_rng(seed=seed)
        sample = rng.uniform(low=rnge[0], high=rnge[1], size=(N, len(m)))
        ys = np.dot(sample, np.array(m).reshape(-1, 1)) + b
        noise = rng.normal(loc=0., scale=scale, size=ys.shape)
        return (sample, ys + noise)

    def write_data(filename, X, y):
        with open(filename, "w") as file:
            xs = [f"x_{n}" for n in range(X.shape[1])]
            header = xs + ["y"]
            writer = csv.writer(file)
            writer.writerow(header)
            for row in np.hstack((X, y)):
                writer.writerow(row)

    m = np.array(m)
    X, y = linear_data_generator(m, b, rnge, N, scale, seed)
    write_data(output_file, X, y)

# Calling the function with example parameters
generate_data(100, [3, 2], 5, 1.0, [-10, 10], 42, 'data.csv')


def load_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        X, y = [], []
        for row in reader:
            X.append([float(num) for num in row[:-1]])
            y.append(float(row[-1]))
        return np.array(X), np.array(y)

def test_model_with_generated_data(filename):
    X, y = load_data(filename)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = ElasticNetModel(lambdas=1.0, l1_ratio=0.5, iterations=1000, learning_rate=0.01)
    results = model.fit(X_scaled, y)
    
    predictions = results.predict(X_scaled)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions, alpha=0.5)
    plt.title('Comparison of Actual and Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # Diagonal line for reference
    plt.grid(True)
    plt.show()
    
    print("Predictions:", predictions)
    print("Actuals:", y)
    return predictions, y
    
predictions, actuals = test_model_with_generated_data('data.csv')
