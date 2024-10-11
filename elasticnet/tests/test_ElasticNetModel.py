import csv
import numpy
from elasticnet.models.ElasticNet import ElasticNetModel


# Testing the model for making predictions using a small test dataset:
def test_predict():
    model = ElasticNetModel(0.5, 0.000001, 0.000001)
    data = []

    # Reading data from the dowloaded CSV file
    with open("small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = numpy.array([[v for k, v in datum.items() if k.startswith('x')] for datum in data], dtype=float)
    y = numpy.array([[v for k, v in datum.items() if k == 'y'] for datum in data], dtype=float)
    results = model.fit(X, y)
    print("Beta:")
    print(results.beta)

    print(results.predict(X))

    # Plotting the Actual vs Predicted values:
    results.getActualVsTrueGraph(X, y)


# Testing the Prediction using the external CSV file with a chosen target column
def test_external_file(name, target):
    # Running the Elastic net model with the given parameters lambda, l1, l2, scaling flag and its feature range
    # The scaling can be turned off if necessary
    model = ElasticNetModel(0.5, 0.000001, 0.000001, True, (-10, 10))
    data = []
    with open(name, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = numpy.array([[v for k, v in datum.items() if k.strip() != target] for datum in data], dtype=float)
    y = numpy.array([[v for k, v in datum.items() if k.strip() == target] for datum in data], dtype=float)
    # Getting the Elastic net model using the feature matrix, X and target vector, Y
    results = model.fit(X, y)
    print("Beta:")
    print(results.beta)
    # You can use the results class to get the actual vs true values graph as well as the residual graph if needed.
    # You can also use the predict method to get the predictions for the graph
    results.getActualVsTrueGraph(X, y)


test_external_file("industry_plant.csv", "PE")