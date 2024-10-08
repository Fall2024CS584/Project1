import numpy as np
from sklearn.linear_model import ElasticNet as SklearnElasticNet
from elasticnet.models.ElasticNet import ElasticNetModel, MyRSquared, MyMSE
from elasticnet.models.data_generators import (
    linear_data_generator1,
    linear_data_generator2,
    nonlinear_data_generator1,
    generate_collinear_data,
    generate_periodic_data,
    generate_higher_dim_data
)

from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def test_linear_data1():
    print("TEST 1: Linear Data with Single Feature")
    alpha = 0.05
    l1_ratio = 0.9
    max_iter = 500
    tol = 1e-5
    learning_rate = 0.05
    range_ = [-5, 5]
    N = 100
    seed = 42

    X, ys = linear_data_generator1(2, 3, range_, N, seed)
    custom_model = ElasticNetModel(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol, learning_rate=learning_rate)
    results_custom = custom_model.fit(X.reshape(-1, 1), ys)
    y_pred_custom = results_custom.predict(X.reshape(-1, 1))

    sklearn_model = SklearnElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol)
    sklearn_model.fit(X.reshape(-1, 1), ys)
    y_pred_sklearn = sklearn_model.predict(X.reshape(-1, 1))

    r2_custom = MyRSquared.calculate(ys, y_pred_custom)
    mse_custom = MyMSE.calculate(ys, y_pred_custom)
    r2_sklearn = r2_score(ys, y_pred_sklearn)
    mse_sklearn = mean_squared_error(ys, y_pred_sklearn)

    print(f"Custom ElasticNet - R²: {r2_custom:.4f}, MSE: {mse_custom:.4f}")
    print(f"scikit-learn ElasticNet - R²: {r2_sklearn:.4f}, MSE: {mse_sklearn:.4f}")

def test_linear_data2():
    print("TEST 2: Linear Data with Multiple Features")
    alpha = 0.3
    l1_ratio = 0.8
    max_iter = 500
    tol = 1e-5
    learning_rate = 0.05
    range_ = [-5, 5]
    N = 100
    seed = 42
    m = np.array([1.5, -2.0])
    b = 5

    X, ys = linear_data_generator2(m, b, range_, N, seed)
    custom_model = ElasticNetModel(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol, learning_rate=learning_rate)
    results_custom = custom_model.fit(X, ys)
    y_pred_custom = results_custom.predict(X)

    sklearn_model = SklearnElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol)
    sklearn_model.fit(X, ys)
    y_pred_sklearn = sklearn_model.predict(X)

    r2_custom = MyRSquared.calculate(ys, y_pred_custom)
    mse_custom = MyMSE.calculate(ys, y_pred_custom)
    r2_sklearn = r2_score(ys, y_pred_sklearn)
    mse_sklearn = mean_squared_error(ys, y_pred_sklearn)

    print(f"Custom ElasticNet - R²: {r2_custom:.4f}, MSE: {mse_custom:.4f}")
    print(f"scikit-learn ElasticNet - R²: {r2_sklearn:.4f}, MSE: {mse_sklearn:.4f}")

def test_nonlinear_data():
    print("TEST 3: Nonlinear Data")
    alpha = 0.1
    l1_ratio = 0.7
    max_iter = 500
    tol = 1e-5
    learning_rate = 0.01
    range_ = [-5, 5]
    N = 100
    seed = 42

    X, ys = nonlinear_data_generator1(0.5, 2, range_, N, seed)
    custom_model = ElasticNetModel(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol, learning_rate=learning_rate)
    results_custom = custom_model.fit(X.reshape(-1, 1), ys)
    y_pred_custom = results_custom.predict(X.reshape(-1, 1))

    sklearn_model = SklearnElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol)
    sklearn_model.fit(X.reshape(-1, 1), ys)
    y_pred_sklearn = sklearn_model.predict(X.reshape(-1, 1))

    r2_custom = MyRSquared.calculate(ys, y_pred_custom)
    mse_custom = MyMSE.calculate(ys, y_pred_custom)
    r2_sklearn = r2_score(ys, y_pred_sklearn)
    mse_sklearn = mean_squared_error(ys, y_pred_sklearn)

    print(f"Custom ElasticNet - R²: {r2_custom:.4f}, MSE: {mse_custom:.4f}")
    print(f"scikit-learn ElasticNet - R²: {r2_sklearn:.4f}, MSE: {mse_sklearn:.4f}")

def test_collinear_data():
    print("TEST 4: Collinear Data")
    alpha = 0.05
    l1_ratio = 0.9
    max_iter = 500
    tol = 1e-5
    learning_rate = 0.05
    range_ = [-5, 5]
    noise_scale = 0.01
    N = 100
    seed = 42

    X, ys = generate_collinear_data(range_, noise_scale, (N, 3), seed)
    custom_model = ElasticNetModel(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol, learning_rate=learning_rate)
    results_custom = custom_model.fit(X, ys.flatten())
    y_pred_custom = results_custom.predict(X)

    sklearn_model = SklearnElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol)
    sklearn_model.fit(X, ys.flatten())
    y_pred_sklearn = sklearn_model.predict(X)

    r2_custom = MyRSquared.calculate(ys.flatten(), y_pred_custom)
    mse_custom = MyMSE.calculate(ys.flatten(), y_pred_custom)
    r2_sklearn = r2_score(ys.flatten(), y_pred_sklearn)
    mse_sklearn = mean_squared_error(ys.flatten(), y_pred_sklearn)

    print(f"Custom ElasticNet - R²: {r2_custom:.4f}, MSE: {mse_custom:.4f}")
    print(f"scikit-learn ElasticNet - R²: {r2_sklearn:.4f}, MSE: {mse_sklearn:.4f}")

def test_periodic_data():
    print("TEST 5: Periodic Data")
    alpha = 0.5
    l1_ratio = 0.8
    max_iter = 500
    tol = 1e-5
    learning_rate = 0.01
    range_ = [-5, 5]
    period = 5
    amplitude = 10
    noise_scale = 0.5
    N = 100
    seed = 42

    X, ys = generate_periodic_data(period, amplitude, range_, noise_scale, N, seed)
    custom_model = ElasticNetModel(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol, learning_rate=learning_rate)
    results_custom = custom_model.fit(X.reshape(-1, 1), ys)
    y_pred_custom = results_custom.predict(X.reshape(-1, 1))

    sklearn_model = SklearnElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol)
    sklearn_model.fit(X.reshape(-1, 1), ys)
    y_pred_sklearn = sklearn_model.predict(X.reshape(-1, 1))

    r2_custom = MyRSquared.calculate(ys, y_pred_custom)
    mse_custom = MyMSE.calculate(ys, y_pred_custom)
    r2_sklearn = r2_score(ys, y_pred_sklearn)
    mse_sklearn = mean_squared_error(ys, y_pred_sklearn)

    print(f"Custom ElasticNet - R²: {r2_custom:.4f}, MSE: {mse_custom:.4f}")
    print(f"scikit-learn ElasticNet - R²: {r2_sklearn:.4f}, MSE: {mse_sklearn:.4f}")

def test_higher_dim_data():
    print("TEST 6: Higher Dimensional Data")
    alpha = 0.5
    l1_ratio = 0.8
    max_iter = 500
    tol = 1e-5
    learning_rate = 0.005
    range_ = [-5, 5]
    noise_scale = 0.5
    N = 100
    seed = 42

    X, ys = generate_higher_dim_data(range_, noise_scale, (N, 3), seed)
    custom_model = ElasticNetModel(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol, learning_rate=learning_rate)
    results_custom = custom_model.fit(X, ys)
    y_pred_custom = results_custom.predict(X)

    sklearn_model = SklearnElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol)
    sklearn_model.fit(X, ys)
    y_pred_sklearn = sklearn_model.predict(X)

    r2_custom = MyRSquared.calculate(ys, y_pred_custom)
    mse_custom = MyMSE.calculate(ys, y_pred_custom)
    r2_sklearn = r2_score(ys, y_pred_sklearn)
    mse_sklearn = mean_squared_error(ys, y_pred_sklearn)

    print(f"Custom ElasticNet - R²: {r2_custom:.4f}, MSE: {mse_custom:.4f}")
    print(f"scikit-learn ElasticNet - R²: {r2_sklearn:.4f}, MSE: {mse_sklearn:.4f}")

# Call each test function to perform the tests
if __name__ == "__main__":
    test_linear_data1()
    test_linear_data2()
    test_nonlinear_data()
    test_collinear_data()
    test_periodic_data()
    test_higher_dim_data()

