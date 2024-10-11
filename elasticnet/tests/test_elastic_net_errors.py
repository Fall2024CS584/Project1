import numpy as np
from elasticnet.models.ElasticNet import ElasticNetModel

def test_elastic_net_errors():
    # Test cases for ElasticNetModel initialization
    try:
        ElasticNetModel(alpha=-1)
    except ValueError as e:
        print("Error 1:", str(e))

    try:
        ElasticNetModel(l1_ratio=2)
    except ValueError as e:
        print("Error 2:", str(e))

    try:
        ElasticNetModel(learning_rate=0)
    except ValueError as e:
        print("Error 3:", str(e))

    try:
        ElasticNetModel(max_iter=0)
    except ValueError as e:
        print("Error 4:", str(e))

    try:
        ElasticNetModel(tol=0)
    except ValueError as e:
        print("Error 5:", str(e))

    # Test cases for input validation
    model = ElasticNetModel()

    try:
        model.fit([1, 2, 3], [1, 2, 3])  # Not numpy arrays
    except TypeError as e:
        print("Error 6:", str(e))

    try:
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2, 3])  # Mismatched lengths
        model.fit(X, y)
    except ValueError as e:
        print("Error 7:", str(e))

    try:
        X = np.array([[1, 2], [3, np.nan]])
        y = np.array([1, 2])
        model.fit(X, y)
    except ValueError as e:
        print("Error 8:", str(e))

    try:
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, np.nan])
        model.fit(X, y)
    except ValueError as e:
        print("Error 9:", str(e))

    # Test case for predict before fit
    try:
        X = np.array([[1, 2], [3, 4]])
        model.predict(X)
    except ValueError as e:
        print("Error 10:", str(e))

    # Test cases for ElasticNetModelResults
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    results = model.fit(X, y)

    try:
        results.predict([1, 2])  # Not a numpy array
    except TypeError as e:
        print("Error 11:", str(e))

    try:
        results.predict(np.array([[1, np.nan]]))
    except ValueError as e:
        print("Error 12:", str(e))

if __name__ == "__main__":
    test_elastic_net_errors()