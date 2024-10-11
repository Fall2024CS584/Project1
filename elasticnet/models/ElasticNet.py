import matplotlib.pyplot as plt
import numpy
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class ElasticNetModel():
    # The init function takes in the hyperparameters for lambda1 from which lambda2 is also calculated, the threshold which the criteria
    # for stopping the gradient descent allowing the model to converge, the learning rate to specify how fast the model learns.
    # The scale is used to allow the user to either scale or not scale their data and the scale range scales all the values
    # between the specified range.
    def __init__(self, lambda1 = 0.5, threshold = 0.000001, learning_rate = 0.000001, scale = False, scale_range = (-10,10)):
        self.lambda1 = lambda1
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.scaler = MinMaxScaler(feature_range=scale_range)
        self.shouldScale = scale

    def fit(self, A, ys):
        # Checks if scaling is required, if it is then scale the data
        if(self.shouldScale):
            # Scales the train data between the specified range
            A = self.scaler.fit_transform(A)
            ys = self.scaler.fit_transform(ys)
        # Initialized the random multidimensional arrays generator
        rng = numpy.random.default_rng()
        # Create a matrix with 1 column and len(A) rows to account for the intercept
        intercept_ones = numpy.ones((len(A), 1))
        # Append the matrix of all ones to the data to account for the intercept
        A = numpy.c_[intercept_ones, A]
        # Get the number of rows and number of columns for the data
        self.N, self.d = A.shape
        if self.N == 0:
            # If there are no rows then raise an error
            raise ValueError("Number of samples cannot be zero.")
        # Set a random staring point for the beta matrix
        self.beta = rng.normal(loc=0, scale=0.01, size=(self.d, 1))
        # Set a beta before complete with zeroes so that we can compare if we have met the required threshold
        self.beta_before = numpy.zeros(shape=(self.d, 1))
        # Check if the required threshold as been satisfied, if not continue looping
        while (numpy.linalg.norm(self.beta - self.beta_before) > self.threshold):
            # Set the beta before to the current beta
            self.beta_before = self.beta
            # Update the weights
            self.beta = self.change_weights(A, ys)
        # Once the beta has converged return a Result Class with the beta value stored in it
        return ElasticNetModelResults(self.beta,self.scaler,self.shouldScale)

    def change_weights(self, A, ys):
        # Create an empty gradient matrix filled with zeroes
        gradient = numpy.zeros_like(self.beta)
        # Get the predictions for the current values of beta using the dot product
        predictions = numpy.dot(A, self.beta)
        # Use the gradient formula for Elastic Net Regression to calculate each of the gradient values and store
        # it in the gradient matrix
        for i in range(self.d):
            if self.beta[i, 0] > 0:
                gradient[i, 0] = (-2 * numpy.dot(A[:, i], (ys - predictions)) + self.lambda1 + (
                            2 * (1 - self.lambda1) * self.beta[i, 0])) / self.d
            elif self.beta[i, 0] < 0:
                gradient[i, 0] = (-2 * numpy.dot(A[:, i], (ys - predictions)) - self.lambda1 + (
                            2 * (1 - self.lambda1) * self.beta[i, 0])) / self.d
            else:
                gradient[i, 0] = (-2 * numpy.dot(A[:, i], (ys - predictions)) + (
                            2 * (1 - self.lambda1) * self.beta[i, 0])) / self.d
        # Apply the learning rate to the gradient and substract it from the beta to move the beta closer to its actual value
        return self.beta - (self.learning_rate * gradient)

class ElasticNetModelResults():

    # Initializing the model's parameters through the constructor function:
    def __init__(self, beta, scaler, shouldScale):
        #   beta: Coefficients of the ElasticNet model (including intercept)
        #   scaler: Scaler object used to scale the features (e.g., StandardScaler or MinMaxScaler)
        #   shouldScale: Boolean indicating whether the features should be scaled before prediction
        self.beta = beta
        self.scaler = scaler
        self.shouldScale = shouldScale

    # Predicting the output using the model's coefficients:
    def predict(self, x):
        # Scaling the features and target values if the shouldScale flag is True
        if (self.shouldScale):
            x = self.scaler.fit_transform(x)

        # Adding a column of ones for the intercept term.
        intercept_ones = numpy.ones((len(x), 1))

        x_b = numpy.c_[intercept_ones, x]
        return numpy.dot(x_b, self.beta)

    # Creating a scatter plot comparing actual vs predicted values:
    def getActualVsTrueGraph(self, x, y):
        if (self.shouldScale):
            x = self.scaler.fit_transform(x)
            y = self.scaler.fit_transform(y)
        intercept_ones = numpy.ones((len(x), 1))
        x_b = numpy.c_[intercept_ones, x]

        # Calculating predicted values
        pred = numpy.dot(x_b, self.beta)

        # Creating a scatter plot with actual values on x-axis and predicted values on y-axis
        plt.scatter(y[:, 0], pred[:, 0], color='green', alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs. Actual Plot')
        plt.show()

    # Creating a residual plot, which visualizes the difference between actual and predicted values:
    def getResidualGraph(self, x, y):
        if (self.shouldScale):
            x = self.scaler.fit_transform(x)
            y = self.scaler.fit_transform(y)
        intercept_ones = numpy.ones((len(x), 1))
        x_b = numpy.c_[intercept_ones, x]
        pred = numpy.dot(x_b, self.beta)

        # Calculating the residuals (differences between actual and predicted values)
        residual = y[:, 0] - pred[:, 0]

        plt.scatter(pred[:, 0], residual, color='blue', alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')

        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()
