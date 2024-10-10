# import csv
import numpy as np
from elasticnet.models.ElasticNet import ElasticNetModel


def generate_synthetic_data(n_samples=50, n_features=3, weights=None, noise_std=0.1, random_state=None, test_ratio=0.2):
    """
    n_samples: Number of samples to generate
    n_features: Number of features to generate
    weights: predefined weights - Create a random weight if no weights are defined
    noise_std: noise scale. default is 0
    random_state: random seed
    sample_weight: portion of the test data over total samples. default is 0.2
    """
    if random_state:
        np.random.seed(random_state)

    test_size = int(n_samples * test_ratio)

    # random sample
    x = np.random.rand(n_samples, n_features)

    # create weighted if not specified
    if weights is None:
        weights = np.random.randn(n_features)

    # create noise
    noise = np.random.normal(0, noise_std, size=n_samples)

    # create y
    y = np.dot(x, weights) + noise

    return x[test_size:], y[test_size:], x[:test_size], y[:test_size], np.array(weights)[:, np.newaxis]


def euclidean_distance(actual_weight, predicted_weight):
    """
    Calculates the Euclidean distance between two weights to check similarity
    :param actual_weight: Predefined weights matrix(Or vector)
    :param predicted_weight: Trained weights matrix(Or vector)
    :return: Computed distance between two matrices (Near 0 will be good!)
    """
    if actual_weight.shape == predicted_weight.shape:
        return np.sqrt(np.sum((actual_weight - predicted_weight) ** 2))


def r2_score(y_true, y_pred):
    """
    Calculates the R2 score between two weights to check the validity of the model.
    :param y_true: Actual weights
    :param y_pred: Predicted weights
    :return: R2 score between two weights
    """
    # Average of actual weights
    y_mean = np.mean(y_true)

    # Total Sum of Squares, TSS
    TSS = np.sum((y_true - y_mean) ** 2)

    # Residual Sum Of Squares, RSS
    RSS = np.sum((y_true - y_pred) ** 2)

    # Compute R2
    r2 = 1 - (RSS / TSS)

    return r2


def test_predict():
    # Test data 1:  small_test.csv
    # data = []
    # with open("small_test.csv", "r") as file:
    #     reader = csv.DictReader(file)
    #     for row in reader:
    #         data.append(row)
    #
    # train_x = np.array([[v for k, v in datum.items() if k.startswith('x')] for datum in data])
    # train_x = np.array([[v for k, v in datum.items() if k == 'y'] for datum in data])
    # test_x = train_x # Set as the same
    # test_y = train_y # Set as the same

    # Test data 2:  Random generation
    # n_samples=50, n_features=3, weights=random, noise_std=0.0
    train_x, train_y, test_x, test_y, true_weights = generate_synthetic_data(n_samples=50, n_features=3)

    # define the parameters
    param = {
        "learning_rate": 0.01,
        "epochs": 1000,
        "alpha": 0.001,
        "rho": 0.015,
        "optimization": True
    }
    model = ElasticNetModel(**param)

    results = model.fit(train_x, train_y)
    predicts = results.predict(test_x)
    print(predicts)
    print("=" * 100)
    print(f"R2 score of the model is {r2_score(test_y, predicts)}")
    print("Number of the best epochs: ", results.epoch)
    print("Cost of the best epochs: ", results.cost)
