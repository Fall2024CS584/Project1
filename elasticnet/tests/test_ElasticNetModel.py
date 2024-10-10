# import csv
import numpy as np
from elasticnet.models.ElasticNet import ElasticNetModel


def generate_synthetic_data(n_samples=50, n_features=3, weights=None, bias=None, noise_std=0.0, random_state=None,
                            test_ratio=0.2) -> tuple:
    """
    generate_synthetic_data()
    This function generates data based on randomly assigned weights and deviations.
    It is designed to verify the robustness of the model by adding the noist_std variable.
    If the model tracks the assigned weights and deviations similarly from that function,
    the model is assumed to have been trained correctly.
    :param n_samples: Number of samples to generate
    :param n_features: Number of features to generate
    :param weights: predefined weights - Create a random weight if no weights are defined
    :param bias: predefined bias - Create a random bias if no bias is defined
    :param noise_std: Set noise scale to test durability of the trained model. default std is 0.0 (= set None)
    :param random_state: random seed
    :param test_ratio: portion of the test data over total samples. default is 0.2
    :returns
        train_x, train_y, test_x, test_y: train and test set
        weights: predefined weights (Includes generated)
        bias: predefined bias (Includes generated)
    """
    # Setup random seed
    if random_state:
        np.random.seed(random_state)

    # random sample
    x = np.random.rand(n_samples, n_features)

    # create weighted if not specified
    if weights is None:
        weights = np.random.randn(n_features)

    # create bias if not specified
    if bias is None:
        bias = np.random.rand()

    # create noise
    noise_std = np.random.normal(0, noise_std, size=n_samples)

    # create y
    y = np.dot(x, weights) + (bias + noise_std)

    # split test and train set
    if 0 < test_ratio < 1:
        test_size = int(n_samples * test_ratio)
        train_x, train_y, test_x, test_y = (
            x[test_size:], y[test_size:, np.newaxis], x[:test_size], y[:test_size, np.newaxis])
    else:  # otherwise set train set == test set (self prediction) * Not recommended
        train_x, train_y, test_x, test_y = x[:], y[:, np.newaxis], x[:], y[:, np.newaxis]

    return train_x, train_y, test_x, test_y, np.array(weights)[:, np.newaxis], bias


def euclidean_distance(ws: np.ndarray, b: float, pws: np.ndarray, pb: float) -> float:
    """
    euclidean_distance()
    This function measures the similarity of two weights (actual, predicted) using the Euclidean distance.
    The more similar it is, the closer it is to 0.
    Accordingly, the model is considered to be a good estimate of the actual function.

    :param ws: Actual weights matrix
    :param b: Actual bias
    :param pws: Predicted weights matrix
    :param pb: Predicted bias
    :return: Computed distance between two matrices (Includes bias)
    """
    try:
        ws = np.insert(ws, 0, b, axis=0)
        pws = np.insert(pws, 0, pb, axis=0)
        return np.sqrt(np.sum((ws - pws) ** 2))
    except Exception:
        return float('inf')  # In case of Error, return infinite


def compute_r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    compute_r2()
    This function calculates the coefficient of determination (R2) between given data (actual y and predicted y).
    The higher the value of R², the higher the explanatory power of the model.
    Formula used:
        R² = 1 - (∑((y_i - y_pred_i)²) / ∑((y_i - y_mean)²))
    where:
        y_i      : actual weight of ith feature
        y_pred_i : predicted weight of ith feature
        y_mean   : mean of the actual weight features

    :param y: Actual weights
    :param y_pred: Predicted weights
    :return: R2 score between two weights
    """
    # Average of actual weights
    y_mean = np.mean(y)
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2)  # R2 = 1 - SSR/SST


def test_predict():
    # STEP 0: Data import/generation

    print("\n", "=" * 100)

    """

    Test data 1:  Data import (e.g. small_test.csv)
    data = []
    with open("small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    train_x = np.array([[v for k, v in datum.items() if k.startswith('x')] for datum in data])
    train_x = np.array([[v for k, v in datum.items() if k == 'y'] for datum in data])
    test_x = train_x # Set as the same
    test_y = train_y # Set as the same

    """

    # """

    # Test data 2:  Random generation (e.f. Use function generate_synthetic_data)

    # User defined params for data creation setting
    data_params = {
        "n_samples": 50,  # Number of samples to create
        "n_features": 3,  # Number of features of the data
        "weights": None,  # Predetermined weights - If set as None, it creates the new one
        "bias": None,  # Predetermined bias - If set as None, it creates the new one
        "noise_std": 0.0,  # Noise scale to check durability of the model
        "random_state": None,  # Random seed
        "test_ratio": 0.2,  # Portion of the test set from total sample.
    }

    print(f"Data Creation Parameters:\n{data_params}")

    # n_samples=50, n_features=3, weights=random, noise_std=0.0
    train_x, train_y, test_x, test_y, weights, bias = generate_synthetic_data(**data_params)

    print(f"Actual weights:\n {weights}")  # pre-determined weight
    print(f"Actual bias: {bias}")  # pre-determined bias
    # """

    print(f"Number of training data samples-----> {train_x.shape[0]}")
    print(f"Number of training features --------> {train_x.shape[1]}")
    print(f"Shape of the target value ----------> {train_y.shape}")

    # STEP 1: Define the parameters

    print("=" * 100)

    # User defined params for training model
    params = {
        "learning_rate": 0.01,  # Learning rate for gradient regression
        "epochs": 10000,  # Number of trainings
        "alpha": 0.0001,  # Strength controller of regularization
        "rho": 0.015,  # L1 ratio
        "optimization": True  # True to activate the best model setting (otherwise, returns the final model)
    }

    print(f"Parameters:\n{params}")

    # STEP 2: Initialize the model

    model = ElasticNetModel(**params)

    # STEP 3: Train the model

    results = model.fit(train_x, train_y)

    # STEP 4: Test the model

    predicts = results.predict(test_x)

    print("=" * 100)
    print(f"Predicted weight:\n{predicts}")

    print(f"R2 score of the model: {compute_r2(test_y, predicts)}")
    print(f"Distance from Actual Parameters: {euclidean_distance(weights, bias, results.w, results.b)}")
    print(f"Number of the epochs: {results.epoch}")
    print(f"Cost of the model: {results.cost}")
