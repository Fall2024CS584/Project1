### Amruta Sanjay Pawar- A20570864 ###
### Raghav Shah- A20570886 ###
### Shreedhruthi Boinpally- A20572883 ###

### How to Run the Code ###
### Command in Terminal: python -m elasticnet.tests.test_ElasticNetModel ###

# Elastic Net Regression Model

## Overview
The present model was developed with the aim of combining the advantages of L1 (Lasso) and L2 (Ridge) penalties in the same framework. The Elastic Net Regression Model is a linear regression model that combines both L1 and L2 regularization methods.

### 1. What does the model you have implemented do and when should it be used?
The Elastic Net regression estimator is a linear model that attempts to wave L1 and L2 regularization over a joint index parameter. The following situations exist:
1. **Multicollinearity**: A few explanatory variables out of the ten are highly correlated and their connection is unstable.
2. **High-dimensional Data**: Through text analysis and other machine learning applications, the large number of variables makes big data analysis both labor- and time-intensive. The use of high-dimensional data is increasing with the development of computational methods, such as bootstrapped standards that detect all instances and prevent erroneous selection of variables, commonly known as 'false positives.'
3. **Feature Selection**: It is easy to change the number of features the model uses and subsequently the target, which in effect may alter the model's final accuracy. However, the extent to which the attributes change generally depends on the distribution of the weather variables. Hence, one cannot absolutely state that the effect will always be the same over the country from which the data was collected.

### 2. How did you test your model to determine if it is working reasonably correctly?
The testing of the model constitutes the following procedures:
1. **Custom Train-Test Split**: Rather than using the libraries already available on scikit-learn, a custom function was created to split the data into training and testing subsets.
2. **Loss Evaluation**: The program monitors and displays training loss per iteration, which is used to determine if the algorithm is converging, and later for evaluating the system’s performance.
3. **Assertions**: Assertions were added to ensure that the predicted values are within a particular tolerance of the actual values, ensuring the model gives correct results during predictions.

### 3. Which of the parameters that are part of your implementation have you exposed to the users in order to get the performance you desire?
Several hyperparameters can be tuned by the model to improve results:
- **alpha**: How much the mean square of the coefficient should be penalized since the regularization terms drive the coefficient values to zero.
- **l1_ratio**: Tracing the effect of Lasso and Ridge on the model through this parameter. Lasso is equivalent to setting the regularization ratio to one, while Ridge has a different penalty.
- **max_iter**: The maximum number of iterations for the optimization algorithm. This can be increased to allow the algorithm to converge, which is useful on difficult datasets.
- **tol**: Tolerance for stopping criteria, which can be adjusted for more precise convergence.

### 4. Is there a particular feature of your implementation that your application could not deal with all the time? If provided with more time, is there any way of solving such issues or is it a part of the model?
Yes, but only in certain cases do the following restrictions apply:
- **Non-Numeric Data**: The current solution assumes that all input features are discrete, thus it will throw errors for commands that are not of the respective class. Input data validation to handle such data is necessary.
- **Convergence Issues**: The model may fail to converge for datasets with severe multicollinearity or high dimensionality within the specified maximum number of iterations. Users may need to tune `max_iter`, `alpha`, and `l1_ratio` for better performance.

#### Possible Improvements:
- Input validation to handle non-numeric or missing data.
- Enabling feature engineering capabilities for interaction or polynomial features.
- Implementing hyperparameter optimization methods such as grid search and random search to automatically identify the most suitable settings.

## Code Explanation

### 1. ElasticNetModel Class
The `ElasticNetModel` class implements Elastic Net regression:
- **Instance Creation**: Instances of the IncrementalLearningMachine type are generated from the learned model and saved in a zip file, with the constructor initializing model parameters (`alpha`, `l1_ratio`, `max_iter`, `tol`).
- **Soft Thresholding**: A function in Lasso regularization, `_soft_threshold` performs a soft threshold operation on the coefficients.
- **Compute Loss**: The `_compute_loss` method calculates total loss as MSE + Regularization Penalties (L1 and L2). It calls `_compute_mse` and `_compute_penalty` methods to compute those components.
- **Fitting the Model**: Normalizes input features, initializes coefficients and intercept, and iteratively updates them until convergence based on the specified tolerance.

### 2. ElasticNetModelResults Class
Results of the fitted model are contained in the `ElasticNetModelResults` class:
- **Predictions**: Scaling is extracted without applying it (fit and transform are called with `.fit`) to scale new data.
- **Loss Visualization**: The loss history during training is plotted for a better understanding of convergence behavior.
- **print_summary**: Returns a neat summary of model performance, including coefficients, intercept, number of iterations, and final loss.

### 3. Model Testing
The test script `test_ElasticNetModel.py` includes:
- **Data Loading**: Reads data from a CSV file and splits it into features and target variables.
- **Custom Train-Test Split**: This function shuffles the data randomly and splits it into training and test sets.
- **Model Fitting and Prediction**: Initializes the model with given hyperparameters, fits the model to the training data, makes predictions on the test data, and evaluates the results.
- **Assertions and Visualization**: Checks whether predictions are within a certain tolerance of actual values, shows loss history, and prints a summary of the model.

## Model Implementation
Below is the implementation of the `ElasticNetModel` class, which implements the Elastic Net regression algorithm and allows users to set the above-mentioned hyperparameters.

## Conclusion
The Elastic Net Regression Model provides a flexible and powerful approach to regression analysis, especially for complex datasets. Users can easily fit and evaluate models while tuning key parameters for optimal performance.
