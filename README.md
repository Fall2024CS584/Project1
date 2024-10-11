# Project 1 - ElasticNet Regularization
Submitted by 
1. Anushka Sarath - A20576979
2. B P Gayathri Ananya - A20588605
3. Gladys Gince Skariah - A20576603

## 1. Introduction 

What does the model you have implemented do and when should it be used?

### 1.1 Model description:
The __Elastic-Net Regression__ is built on the Linear Regression which shares the same hypothetical function for prediction
```y = weight*x + bias```
It is a combination of Lasso Regression which adds the L1 regularization penalty and Ridge regression which adds the L2 regularization penalty. 
One of the main disadvantages of linear regression is overfitting and it cannot deal with collinear data. When the dataset used has more observations and a lesser number of features it provides an opportunity for the model to make highly accurate predictions thereby overfitting the data. Such a model tends to have a high variance when the test data is passed and does not generalize on the new data. 
To overcome this limitations, we include both L-1 and L-2 regularization to get the benefits of both Ridge and Lasso. The resultant model has better predictive power and performs feature selection.
The modified cost function for ElasticNetRegularization is 

Click the Image to view
<div style="background-color: #ffffff; padding: 10px; border: 1px solid black; display: inline-block;">
  <img src="https://quicklatex.com/cache3/6a/ql_5aac9fc5fd4bf23460612319c79a6d6a_l3.png" alt="Click me" />
</div>

where
* Lambda_1 is the L1 penalty including the Lasso regularization
* Lambda_2 is the L2 penalty including the Ridge regularization
* ypred_i is the predicted value (hypothesis) for the i-th training example

Elastic net is mainly used when we don't know whether all the features have significance and there is a strong correlations between features.
To optimize the model and determine the best weight and bias of the model which will minimize the cost function we are using the gradient descent algorithm. This algorithm is called for the number of iterations provided by the user. 

The formula used for the gradient descent:

Weight = weight – learning_rate * variance in cost function wrt variance in weight (dj/dw)

Bias = bias – learning_rate * variance in cost function wrt variance in bias (dj/dw)

where

Click the Image to view
<div style="background-color: #ff0000; padding: 10px; border: 1px solid black; display: inline-block;">
  <img src="https://quicklatex.com/cache3/f2/ql_107e5fc8983b885957fa0292cb49aaf2_l3.png" alt="Click me" />
</div>

<div style="background-color: #ff0000; padding: 10px; border: 1px solid black; display: inline-block;">
  <img src="https://quicklatex.com/cache3/f2/ql_107e5fc8983b885957fa0292cb49aaf2_l3.png" alt="Click me" />
</div>

### 1.2 Implementation

#### Class created: 
1.	Class ElasticNetModel()
2.	Class ElasticNetModelResults()

#### Functions created within the class ElasticNetModel():

1. __init__() :
   * Input parameters: alpha, l1_ratio, learning rate, max_iter, tol
   * Function is used to initialize all the parameters used and also performs basic validations on these parameters to ensure the correct values are being passed for successful execution
       * Alpha: regularization strength.
           i. A higher value of alpha increases the strength of the regularization making the model more biased but less likely to overfit
           ii. lower the alpha values reduces the regularization, allowing the model to fit the data more closely but increasing the risk of overfitting.
           iii. Default value used 1.
       * L1_ratio: determines the balance between Lasso and Ridge regression. Accepted values are 
            i.	0 – model uses purely L2 regularization, which focuses on retaining all the weights of the features within a specified magnitude
            ii.	1 – model uses purely L1 regularization, which focuses on reducing the weights of the less important features to nearly 0.
            iii.	0 > l1_ratio < 1: model combines both L1 and L2 regularization
            iv.	Default value used: 0.5 
       * Learning_rate: the amount of change that needs to be applied to the models co-efficient to minimize the cost function. This is used in the gradient descent function. The default value is set to 0.01
       * max_iter: number of times the optimization technique will run through the data to minimize the loss function. Default value is set to 1000
       * tol: it controls how small the updates to the model coefficients need to be for the algorithm to stop early. Default value is set to 1e-4

2. _validate_input():
    * Input parameters: X,Y
    * The function is used to perform basic validation checks to see if the input data is acceptable
    * Constraints applied:
        * X and Y should be numpy arrays
        * The number of samples in X should be equaly to the number of samples in Y
        * X and Y should not have null or undefined values

3. fit():
    * Input parameters: X, Y
    * The function fits the  model with the training data set. It also validates the input data. The gradient descent optimization algorithm is used in the function to minimise the cost function. It raises an error if the predicted values are undefined as part of the optimization algorithm.
    * The function passes the optimized coefficients and intercept value to the ElasticNetModelResults() where it is used to predict the data.
  
#### Functions used in ElasticNetModelResults():

1.	__init__():
    * Input parameters: coefficient, intercept
    * Initializes the parameters which is received from the fit function

2.	__Predict()__:
    * This function is used to predict the values
    * As elastic net regularization is built on the Linear regression model we are using the linear equation y=mx+c to predict the values
    * The function returns the predicted values.
      
### 1.3 Steps to run the code

## Activating the Virtual Environment

1. **For Windows**:
   - Navigate to the directory where the virtual environment is located:
     ```bash
     cd /path/to/your/project
     ```
   - Activate the virtual environment:
     ```bash
     .\new_env\Scripts\activate
     ```
2. **For macOS/Linux**:
   - Navigate to the directory where the virtual environment is located:
     ```bash
     cd /path/to/your/project
     ```
   - Activate the virtual environment:
     ```bash
     source new_env/bin/activate
     ```
Once the environment is activated, you can proceed to install the necessary dependencies and run the scripts.

## Running the Test Files

* **To test using the California Housing dataset:**
   ```bash
   python -m elasticnet.tests.test_california_housing

* **To test using the Netflix Stock Prediction dataset:**
  ```bash
   python -m elasticnet.tests.test_california_housing

* **To test using the Small test dataset:**
  ```bash
   python -m elasticnet.tests.small_test

* **To test using the Synthetic dataset:**
  ```bash
   python -m elasticnet.tests.test_synthetic_data

## 2. Testing
How did you test your model to determine if it is working reasonably correctly?

### 2.1 Test case 0: 
Validating all the input variables to see if they have the expected values before we fit the model.

* alpha must be a non-negative float or integer

__Input__ :-

![Project Screenshot](elasticnet/images/1_alpha_input.png "Screenshot of the Project")

__Output__ :- 

![Project Screenshot](elasticnet/images/1_alpha_output.png "Screenshot of the Project")

* l1_ratio must be a float between 0 and 1

__Input__ :-

![Project Screenshot](elasticnet/images/2_l1_ratio_input.png "Screenshot of the Project")

__Output__: :- 

![Project Screenshot](elasticnet/images/2_l1_ratio_output.png "Screenshot of the Project")

* learning_rate must be a positive float

__Input__ :-

![Project Screenshot](elasticnet/images/3_learning_rate_input.png "Screenshot of the Project")

__Output__ :- 

![Project Screenshot](elasticnet/images/3_learning_rate_output.png "Screenshot of the Project")

* max_iter must be positive integer

__Input__ :-

![Project Screenshot](elasticnet/images/4_max_iter_input.png "Screenshot of the Project")

__Output__ :- 

![Project Screenshot](elasticnet/images/4_max_iter_output.png "Screenshot of the Project")

* tol must be a positive float

__Input__ :-

![Project Screenshot](elasticnet/images/5_tol_input.png "Screenshot of the Project")

__Output__ :- 

![Project Screenshot](elasticnet/images/5_tol_output.png "Screenshot of the Project")

* X must be a numpy array

__Input__ :-

![Project Screenshot](elasticnet/images/6_X_input.png "Screenshot of the Project")

__Output__ :- 

![Project Screenshot](elasticnet/images/6_X_output.png "Screenshot of the Project")

* X and y must have the same number of samples

__Input__ :-

![Project Screenshot](elasticnet/images/7_sample_input.png "Screenshot of the Project")

__Output__ :- 

![Project Screenshot](elasticnet/images/7_sample_output.png "Screenshot of the Project")

* X contains NaN values

__Input__ :-

![Project Screenshot](elasticnet/images/8_nan_input.png "Screenshot of the Project")

__Output__ :- 

![Project Screenshot](elasticnet/images/8_nan_output.png "Screenshot of the Project")

* y contains NaN values

__Input__ :-

![Project Screenshot](elasticnet/images/9_y_input.png "Screenshot of the Project")

__Output__ :- 

![Project Screenshot](elasticnet/images/9_y_output.png "Screenshot of the Project")

* Model is not fitted yet, ensure we call fit with appropriate arguments before using 'predict'

__Input__ :-

![Project Screenshot](elasticnet/images/10_model_input.png "Screenshot of the Project")

__Output__ :- 

![Project Screenshot](elasticnet/images/10_model_output.png "Screenshot of the Project")

### 2.2 Test case 1:
__small_test.csv__ 
1. __Data loading and preprocessing__: The test The script loads the dataset (small_test.csv), extracts the feature columns (starting with x) and the target column (y), and standardizes the features using StandardScaler to improve model performance.
2. __Model Training and Prediction__: An ElasticNet model is trained on the standardized training data, and predictions are made on both the training and test sets.
3. __Performance Evaluation__: The script calculates and prints key evaluation metrics such as Mean Squared Error (MSE) and R-squared for both the training and testing data, giving insights into the model’s accuracy.
4. __Visualization__: A scatter plot is generated to compare the actual vs predicted values for the test set, using one of the selected features from the dataset on the x-axis.
5. __Dataset__: The dataset contains a set of features labelled as x1, x2, etc., and a target variable y that we are trying to predict. The goal is to assess the performance of the ElasticNet model in predicting y based on the input features.
6. Parameters set:
    * alpha : 35
    * l1_ratio : 0.7
    * learning_rate : 0.001
    * max_iter : 5000
    * tol : 1e-5

The R2 and MSE given by the model for the small_test.csv dataset

![Project Screenshot](elasticnet/images/small_test_evaluation.jpeg "Screenshot of the Project")

plot comparing the actual and predicted values: 

![Project Screenshot](elasticnet/images/small_test_plot.jpeg "Screenshot of the Project")


### 2.3 Test case 2:
__synthetic dataset__: 
1. Synthetic Data Generation: The test generates a synthetic dataset with 13,000 samples and 35 features, simulating real-world data with some added noise. Each feature is randomly generated, and the target variable y is calculated based on a set of randomly generated true coefficients.
2. Data Preprocessing: The synthetic data is split into training and testing sets, and the features are standardized using StandardScaler to improve model performance.
3. Model Training and Prediction: An ElasticNet model is trained on the standardized training data, and predictions are made on the test data.
4. Evaluation and Visualization: The model’s performance is evaluated using Mean Squared Error (MSE) and R-squared metrics, and a scatter plot is created to compare actual vs predicted values based on one selected feature.
5.  Parameters set:
    * alpha : 45
    * l1_ratio : 0.4
    * learning_rate : default
    * max_iter : default
    * tol : default

The R2 and MSE given by the model for the synthetic dataset

![Project Screenshot](elasticnet/images/synthetic_evaluation.jpeg "Screenshot of the Project")

plot comparing the actual and predicted values: 

![Project Screenshot](elasticnet/images/synthetic_plot.jpeg "Screenshot of the Project")

### 2.4 Test case 3: 
__California housing dataset__ : The California Housing dataset contains features like longitude, latitude, housing_median_age, and median_income, describing housing and demographic information. The goal is to build a predictive model to estimate median_house_value, offering insights into how housing characteristics affect prices in California.
1. Data Preprocessing - The test class loads the California Housing dataset, handles missing values by filling with the mean, standardizes numerical features, and splits the data into training and test sets.
2. Model Training - An ElasticNet model is initialized with specific hyperparameters, trained on the scaled training data, and used to make predictions for both the training and test sets.
3. Model Evaluation - The test class calculates key performance metrics such as Mean Squared Error (MSE) and R-squared for both training and test data, helping evaluate the model’s accuracy.
4. Visualization-
    * The test class generates a scatter plot comparing actual vs predicted house values against a selected feature, providing insight into the model's performance for that feature.
    * The heatmap in the notebook is a correlation matrix that shows the strength of relationships between features in the California Housing dataset, highlighting how features like median_income and total_rooms are correlated with each other and with median_house_value, helping to identify key features for model improvement.
5. Parameters set:
    * alpha : 30
    * l1_ratio : 0.7
    * learning_rate : default
    * max_iter : default
    * tol : default

The R2 and MSE given by the model for the california housing dataset:

![Project Screenshot](elasticnet/images/california_evaluation.png "Screenshot of the Project")

Heatmap showing us the correlation amongst the features:

![Project Screenshot](elasticnet/images/california_plot_1.png "Screenshot of the Project")

Plot comparing the actual and predicted values: 

![Project Screenshot](elasticnet/images/california_plot_2.jpeg "Screenshot of the Project")

### 2.5 Test case 4:
__Netflix stock prediction dataset__: 
1. Loading the Netflix Stock Dataset: The Netflix dataset includes stock-related features such as Open, High, Low, Close, Adjusted Close prices, and Volume, with the aim of predicting the next day's Close price using these features.
2. Feature Engineering and Data Preprocessing: It creates additional technical indicators (returns, moving averages, price ranges, etc.) and prepares the data for model training by scaling the features and splitting them into training and testing sets.
3. Training the ElasticNet Model: An ElasticNet regression model is trained on the training data to predict the next day's closing price (Target). The model’s parameters (alpha, l1_ratio, learning_rate) are fine-tuned for stock prediction tasks.
4. Evaluating Model Performance: The model's performance is evaluated using Mean Squared Error (MSE) and R-squared metrics for both training and test sets. A plot is also generated to compare actual vs predicted stock prices using the Close price as the primary feature.
5. Correlation matrix- The correlation matrix shows that all price-related features (Open, High, Low, Close, Adjusted Close) are strongly positively correlated, while Volume has a slight negative correlation with these prices.
6. Parameters set:
    * alpha: 40
    * l1_ratio : 0.6
    * learning_rate : default
    * max_iter : 2000
    * tol : default
      
The R2 and MSE given by the model for the netflix stock prediction dataset:

![Project Screenshot](elasticnet/images/netflix_evaluation_metrics.jpeg "Screenshot of the Project")

Plot comparing the actual and predicted values: 

![Project Screenshot](elasticnet/images/netflix_plot_1.jpeg "Screenshot of the Project")

Heatmap showing us the correlation amongst the features:

![Project Screenshot](elasticnet/images/netflix_plot_2.jpeg "Screenshot of the Project")

## 3. Parameters to tune performance:
What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)

The parameters available for the users which they can use to tune the performance are
1. Alpha : Default value used 1.
2. l1_ratio : 0 > l1_ratio < 1: model combines both L1 and L2 regularization
3. learning_rate : default value is set to 0.01
4. max_iter : default value is set to 1000
5. tol: default set 1e-4

## 4. Future scope
Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

1. Missing or NaN values: In the implementation, we have included checks for NaN values, and if any are found in the dataset (X or y), a ValueError is raised, causing the execution to stop. This can be a problem when dealing with real-world data, as missing values are quite common. However, with more time, we could explore more robust methods that don't require fully complete data. These issues aren't fundamental to the algorithm itself but relate to data handling, which can be addressed with additional preprocessing.
2. Non-numerical data: In my implementation, the code assumes that X is a numerical numpy array, so if the dataset contains categorical or non-numerical features, it will fail. Given more time, we could address this by adding preprocessing steps, like one-hot encoding, to handle categorical data properly. This is not a fundamental issue with the algorithm itself but a necessary data preprocessing step that would ensure the model can work with a wider range of datasets.
3. Outliers: Having extreme values in the data (like very high or low numbers), they messed up the ElasticNet model’s predictions since the model is sensitive to such values.

