from elasticnet.models.ElasticNet import *
from elasticnet.tests.lib import *


def test_predict():
    # STEP 0: Data import/generation [data options] 'file' ,'multi_collinear', 'synthetic'(default)
    # Please set data type to the 'data_selection' variable located in 'elasticnet/tests/params.py'
    # If no value, or invalid option has been set, then it automatically chooses 'synthetic'

    print()
    print("=" * 100)

    # Call data
    train_x, train_y, test_x, test_y, weights, bias = get_data()

    if weights is not None and bias is not None:
        print(f"Actual weights:\n {weights}")  # pre-determined weight
        print(f"Actual bias: {bias}")  # pre-determined bias

    print(f"Number of training data samples-----> {train_x.shape[0]}")
    print(f"Number of training features --------> {train_x.shape[1]}")
    print(f"Shape of the target value ----------> {train_y.shape}")

    # STEP 1: Define the parameters

    print("=" * 100)

    params = get_params()  # pre-set parameters in form as dictionary

    print(f"Parameters:\n{params}")

    # STEP 2: Initialize the model
    model = ElasticNetModel(**params)

    # STEP 3: Train the model
    model.fit(train_x, train_y)

    # STEP 4: Test the model
    predicts = ElasticNetModelResults(model).predict(test_x)

    print("=" * 100)
    print(f"Predicted values:\n{predicts}")
    print(f"R2 score of the model: {compute_r2(test_y, predicts)}")
    if weights is not None and bias is not None:
        print(f"Distance from Actual Parameters: {euclidean_distance(weights, bias, model.w, model.b)}")
    print(f"Number of the epochs: {model.epoch}")
    print(f"Cost of the model: {model.cost}")
