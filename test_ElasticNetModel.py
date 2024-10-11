import os
import csv
import numpy as np
import pytest
from src.ElasticNet import ElasticNetModel
from generate_regression_data import linear_data_generator

def load_small_test_data(filename="small_test.csv"):
    """
    Load data from small_test.csv and return feature matrix X and target vector y.
    """
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, filename)
    
    data = []
    try:
        with open(data_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        pytest.fail(f"Test data file not found at {data_path}")

    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])

    return X, y


def generate_synthetic_data(n_samples=100,n_features=10,noise=0.1,seed=42):
    """
    Generate synthetic regression data using the linear_data_generator function.
    """
    np.random.seed(seed)
    m=np.random.randn(n_features)
    b=np.random.randn()

    rnge=(-10,10)

    scale=noise

    X,y=linear_data_generator(m,b,rnge,n_samples,scale,seed)

    return X,y


@pytest.fixture(scope="module")
def small_test_dataset():

    """
    Fixture to provide small_test.csv data.
    """

    return load_small_test_data()


@pytest.fixture(scope="module")
def synthetic_test_dataset():

    """
    Fixture to provide synthetic regression data.
    """

    return generate_synthetic_data()


def test_elasticnet_fit_predict_small_test(small_test_dataset):

    """
    Test the fit and predict methods of ElasticNetModel using small_test.csv.
    """

    X,y=small_test_dataset
    print(f"Small Test Data: X shape {X.shape}, y shape {y.shape}")


    model=ElasticNetModel(alpha=0.1,l1_ratio=0.5,fit_intercept=True,max_iter=10000,tolerance=1e-6,
                          learning_rate=0.05,
                          optimization='batch',random_state=101)


    model.fit(X,y)


    y_pred=model.predict(X)


    mse=np.mean((y-y_pred)**2)


    r2=1-mse/np.var(y)


    assert mse<10,f"MSE is too high: {mse}"


    assert r2>0.8,f"R-squared is too low: {r2}"


def test_elasticnet_fit_predict_synthetic(synthetic_test_dataset):

    """
    Test the fit and predict methods of ElasticNetModel using synthetic regression data.
    """

    X,y=synthetic_test_dataset
    print(f"Synthetic Test Data: X shape {X.shape}, y shape {y.shape}")


    model=ElasticNetModel(alpha=0.1,l1_ratio=0.5,fit_intercept=True,max_iter=10000,tolerance=1e-6,
                          learning_rate=0.5,
                          optimization='batch',random_state=101)


    model.fit(X,y)


    y_pred=model.predict(X)


    mse=np.mean((y-y_pred)**2)


    r2=1-mse/np.var(y)


    assert mse<10,f"MSE is too high: {mse}"


    assert r2>0.8,f"R-squared is too low: {r2}"


def test_zero_variance_feature_small_test(small_test_dataset):

    """
    Test the model's ability to handle a zero variance feature using small_test.csv.
    """

    X,y=[array.copy()for array in small_test_dataset]


    # Introduce a zero variance feature by setting the first feature to a constant


    X[:,0]=5.0


    model=ElasticNetModel(alpha=0.05,l1_ratio=0.5,fit_intercept=True,max_iter=10000,tolerance=1e-9,
                          learning_rate=0.05,
                          optimization='batch',random_state=101)


    model.fit(X,y)


    y_pred=model.predict(X)


    mse=np.mean((y-y_pred)**2)


    r2=1-mse/np.var(y)


    assert mse<10,f"MSE is too high with zero variance feature: {mse}"


    assert r2>0.8,f"R-squared is too low with zero variance feature: {r2}"


def test_zero_variance_feature_synthetic(synthetic_test_dataset):


    """
    Test the model's ability to handle a zero variance feature using synthetic data.
    """


    X,y=[array.copy()for array in synthetic_test_dataset]


    # Introduce a zero variance feature by setting the second feature to a constant


    X[:,1]=10.0


    model=ElasticNetModel(alpha=0.1,l1_ratio=0.5,fit_intercept=True,max_iter=10000,tolerance=1e-6,
                          learning_rate=0.05,
                          optimization='batch',random_state=101)


    model.fit(X,y)


    y_pred=model.predict(X)


    mse=np.mean((y-y_pred)**2)


    r2=1-mse/np.var(y)


    assert mse<10,f"MSE is too high with zero variance feature: {mse}"


    assert r2>0.8,f"R-squared is too low with zero variance feature: {r2}"


def test_invalid_optimization_option_small_test(small_test_dataset):


    """
    Test that providing an invalid optimization algorithm raises a ValueError using small_test.csv.
    """


    X,y=[array.copy()for array in small_test_dataset]


    with pytest.raises(ValueError):

        model=ElasticNetModel(alpha=0.1,l1_ratio=0.5,fit_intercept=True,max_iter=
        10000,tolerance=
                              1e-6,
                              learning_rate=
                              0.01,
                              optimization=
                              'invalid_option',random_state=
                              10)
        model.fit(X,y)


