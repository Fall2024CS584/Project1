import numpy as np
import csv
from params import *

# Collection of auxiliary functions

"""
    Data import/create functions
"""


def read_csv(file_name: str, test_ratio=0.2) -> tuple:
    """
    read_data()
    This function reads the data from the given csv file
    :param file_name: name of the file to read
    :param test_ratio: the ratio of test date (if not in (0,1) then automatically use train data as test data
    :return:
    """
    data = []
    with open(file_name, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    x = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([[float(v) for k, v in datum.items() if k == 'y'] for datum in data])
    print(x.shape,y.shape)
    # split test and train set
    if 0 < test_ratio < 1:
        test_size = int(x.shape[0] * test_ratio)
        train_x, train_y, test_x, test_y = (
            x[test_size:], y[test_size:], x[:test_size], y[:test_size])
    else:  # otherwise set train set == test set (self prediction) * Not recommended
        train_x, train_y, test_x, test_y = x[:], y[:], x[:], y[:]

    return train_x, train_y, test_x, test_y, None, None


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


def generate_multi_collinear_data(n_samples=50, n_features=3, weights=None, bias=None, correlation=0.9, noise_std=0.01,
                                  random_state=None, test_ratio=0.2) -> tuple:
    """
    generate_multi_collinear_data()
    This is a function that generates data with multi_collinearity.
    The data produced from this function is intended to measure the ElasticNet model's ability
    to handle multi_collinearity, which is a critical feature of the model.
    :param n_samples: Number of samples to generate
    :param n_features: Number of features to generate
    :param weights: predefined weights - Create a random weight if no weights are defined
    :param bias: predefined bias - Create a random bias if no bias is defined
    :param correlation: Correlation coefficient between features (1 is the worst)
    :param noise_std: Set noise scale to test durability of the trained model. default std is 0.0 (= set None)
    :param random_state: Set random number seed
    :param test_ratio: portion of the test data over total samples. default is 0.2
    :returns:
        train_x, train_y, test_x, test_y: train and test set
        weights: predefined weights (Includes generated)
        bias: predefined bias (Includes generated)
    """
    if random_state:
        np.random.seed(random_state)

    # random base sample (1st feature
    x_base = np.random.rand(n_samples, 1)

    # Create another feature based on the data for the first feature
    x = x_base + correlation * np.random.randn(n_samples, n_features) * noise_std

    # Increase the correlation between each feature (e.g. through linear combination)
    for i in range(1, n_features):
        x[:, i] = correlation * x_base[:, 0] + (1 - correlation) * np.random.randn(n_samples)

    # create weighted if not specified
    if weights is None:
        weights = np.random.randn(n_features)

    # create bias if not specified
    if bias is None:
        bias = np.random.rand()

    # create y (Add noise to multicollinearity)
    y = x.dot(weights) + bias + noise_std * np.random.randn(n_samples)

    # split test and train set
    if 0 < test_ratio < 1:
        test_size = int(n_samples * test_ratio)
        train_x, train_y, test_x, test_y = (
            x[test_size:], y[test_size:, np.newaxis], x[:test_size], y[:test_size, np.newaxis])
    else:  # otherwise set train set == test set (self prediction) * Not recommended
        train_x, train_y, test_x, test_y = x[:], y[:, np.newaxis], x[:], y[:, np.newaxis]

    return train_x, train_y, test_x, test_y, np.array(weights)[:, np.newaxis], bias


def get_data() -> tuple:
    """
    get_data()
    This function loads training and test from chosen data type(high order function)
    [data options] 'file' ,'multi_collinear', 'synthetic'(default)

    :return: tuples of data from executed functions
    """
    data_type = data_selection

    print(f"Data Type: {data_type}")

    if data_type == 'file':
        print(f"Data Parameters:\n{data_file_params}")
        return read_csv(**data_file_params)
    elif data_type == 'multi_collinear':
        print(f"Data Parameters:\n{data_file_params}")
        return generate_multi_collinear_data(**data_mul_params)
    print(f"Data Parameters:\n{data_file_params}")
    return generate_synthetic_data(**data_syn_params)  # default is normal file generation


def get_params() -> dict:
    """
    get_params()
    This function returns a pre-configured training parameters in form of a dictionary
    :return: test parameters in dictionary format
    """
    return test_params


"""
    Metric functions
"""


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
