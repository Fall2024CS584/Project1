ElasticNet: Custom Elastic Net Regression Model


Nidhi Shrivastav A20594009 nshrivastav@hawk.iit.edu
Rutuja Jadhav A20539073 rjadhav4@hawk.iit.edu
Pankaj Jagtap A20543260 pjagtap1@hawk.iit.edu


A.	Overview:
This project implements a custom Elastic Net regression model in Python, combining L1 (Lasso) and L2 (Ridge) regularization techniques to balance feature selection and coefficient shrinkage. The Elastic Net model is particularly effective for handling multicollinearity (highly correlated features) and sparse features, making it well-suited for various regression tasks. The model is trained through iterative updates of its coefficients using both penalties. Additionally, the project includes methods for predicting target values, evaluating the model’s performance using metrics like MSE, MAE, and R-squared, and visualizing results.

B.	Files
•	ElasticNet.py: Contains the implementation of the ElasticNet class for Elastic Net regression.
•	test_ElasticNetModel.py: Contains unit tests for validating the functionality of the ElasticNet class using pytest.

C.	ElasticNet Class Details
The ElasticNet class implements an Elastic Net regression model that combines L1 and L2 regularization techniques. Here is a detailed breakdown of the functions and parameters involved:

1.	__init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4)
• Purpose: Sets up the Elastic Net model with specified regularization parameters.
•	Parameters:
  - alpha: It determines the magnitude of the regularization.
  - l1_ratio: Relative weight of L1 (Lasso) and L2 (Ridge) regularization.
  - max_iter: Max No. of iterations for the optimization process.
  - tol: Stopping criteria tolerance which is based on change in coefficients.

2.	fit(self, X, y)
•	Purpose: Trains the Elastic Net model using the given features (X) and target values (y).
•	Parameters:
  - X: Input matrix consisting of data (2D array).
  - y: The target values (1D array).
•	Validation Checks:
-	Ensure X is a 2D array and y is a 1D array.
-	Check if the number of samples in X matches the number of samples in y.
-	Making confirm that feature do not 0 variance.
•	Key Operations:
  - Handles regularization through iterative coefficient updates.
  - Incorporates both L1 and L2 penalties when updating coefficients.
  - Stops when the change in coefficients is below a set tolerance.
3.	predict(self, X)
•	Purpose: Predictions on input data
•	Parameters:
  - X: Input data for which predictions are to be made.
•	Returns: Computes the dot product of the input features and the model's coefficients, adding the intercept to return the predicted values based on the model's learned coefficients.

4.	evaluate(self, X, y_true)
• Purpose: Assesses the model's performance by computing the following metrics:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R-squared (R²)
•	Parameters:
  - X: 2D array of input features for evaluation.
  - y_true: 1D array with target values
•	Returns: The calculated MSE, MAE, and R-squared values.

5.	plot_predictions(self, y_true, y_pred)
•	Parameters:
-	y_true: 1D array of actual target values.
-	y_pred: 1D array of predicted target values.

•	Purpose: Plots a scatter plot comparing the actual vs predicted values with a line representing a perfect fit.
6.	plot_residuals(self, y_true, y_pred)
•	Purpose: Plots residuals (differences between actual and predicted values).



Linear Data Generator:

1.	linear_data_generator(m, b, rnge, N, scale, seed)
•	Purpose: Generates random linear data for testing purposes, with noise added.
•	Parameters:
-	 n_samples: Number of samples to generate.
-	n_features: No. features per sample
-	noise: Standard deviation of the noise added to the target values.
-	random_state: Seed for reproducibility.Returns: Generated sample data and noisy target values.
Usage
1.	Testing: The test_ElasticNetModel.py file includes several unit tests to verify the correctness of the ElasticNet class.
2.	Setup: pytest is used to test the model. 
3.	Install pytest via pip: pip install pytest
4.	Run Tests
Go to elasticnet folder on command line/terminal
Use this command to run: 
pytest tests/test_ElasticNetModel.py
5.	Test Functions:

a.	test_fit_and_predict
•	Purpose: Verifies that the model can correctly fit data and make accurate predictions.
•	Test Input: Uses a simple dataset with known target values.
•	Expected Output: Predictions should closely match the target values.

b.	test_invalid_input_shape
•	Purpose: Ensures that a ValueError is raised when the number of samples in X and y do not match.
•	Test Input: X with 2 samples and y with 1 sample.
•	Expected Output: A `ValueError` with a specific error message.

c.	test_no_variance
•	Purpose: Tests the model's behavior when the input features have no variance.
•	Test Input: Data where all feature values are constant.
•	Expected Output: Model should still make predictions and not fail.

d.	test_single_feature
•	Purpose: Ensures the model can handle cases where only one feature is provided.
•	Test Input: A one-dimensional X and target y.
•	Expected output: Ensure that predictions match the target values by checking with np.testing.assert_almost_equal.

e.	test_fit_convergence()
•	Purpose: To test the model’s robustness and convergence on larger datasets.
•	Test Input: random data for X and noisy y values.
•	Expected output: Calculate metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (evaluate function). Check that MSE is below a reasonable threshold (less than 1.0), indicating successful convergence.

f.	test_multiple_coefficients()
•	Purpose: To test the model's ability to fit and predict when multiple features are present.
•	Test Input: data that has two features
•	Expected output: Ensure that the predictions match the expected target values by comparing with np.testing.assert_almost_equal.



Questions

1. What does the model you have implemented do and when should it be used?
The model used is an ElasticNet regression, a type of linear regression that combines L1 regularization (Lasso) and L2 regularization (Ridge). This is helpful when there are many features involved, that too having many correlations. It helps prevent overfitting by introducing a penalty on the coefficients, making the model more generalizable.   
 Used for:  
•	High dimensional data: When the number of features (predictors) is large compared to the number of data points.
•	Multicollinearity: When some features are highly correlated with each other, the ElasticNet regularization can handle this better than a plain linear regression.
•	Feature selection and shrinkage: ElasticNet can help reduce the dimensionality of the problem by zeroing out some coefficients (feature selection) while keeping the others.

2. How did you test your model to determine if it is working reasonably correctly?
To ensure the ElasticNet regression model works correctly, several unit tests were conducted using pytest. These tests evaluated the model's ability to fit data and make accurate predictions, checked for appropriate error handling when input shapes were mismatched, and assessed the model's behavior with edge cases like features with no variance or single-feature data. Additionally, the model's performance was tested on larger datasets to verify convergence and was evaluated using metrics such as mean squared error (MSE), mean absolute error (MAE), and R-squared. These tests confirmed the model's reliability across various scenarios.

3. What parameters have you exposed to users of your implementation in order to tune performance?
The ElasticNet model, like other regularized linear models, has two important parameters that help tune performance:
i.	Alpha (also known as regularization strength):  This parameter controls the total amount of regularization applied to the model. A higher alpha increases the regularization effect, shrinking the model's coefficients more and helping to prevent overfitting. Lower alpha values make the model behave more like a regular linear regression without regularization.
ii.	L1_ratio:  This parameter determines the balance between Lasso (L1) and Ridge (L2) penalties. A value of 1 applies only Lasso (L1), 0 applies only Ridge (L2), and any value in between represents a combination of both regularizations.

4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental to the model?
•	Zero variance in features: If one or more features in the input dataset have no variance (i.e., all values are the same), this can cause issues with the model as the regularization mechanism assumes variability in the input features. Such features can be dropped or handled separately.
•	Multicollinearity: While ElasticNet can handle multicollinearity better than regular linear regression, extremely high correlations between predictors might still pose a challenge, as it can cause instability in the coefficient estimates.
•	Very sparse data: If the dataset is too small or the number of features is much larger than the number of samples, the model may have difficulty finding meaningful patterns.

In many cases, such problems can be mitigated. For example:
•	Feature selection: Features with zero variance can be removed beforehand.
•	Handling multicollinearity: If the correlation between features is too high, techniques like Principal Component Analysis (PCA) or feature scaling can be applied to reduce collinearity.
•	Data augmentation: In cases of sparse data, more data can be collected, or data imputation techniques could be employed.