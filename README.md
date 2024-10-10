# Project 1: Linear Regression with ElasticNet Regularization

## Group Members :

#### •	Kaustubh Dangche - A20550806
#### •	Anu Singh - A20568373
#### •	Hyunsung Ha - A20557555
#### •	Nam Gyu Lee - A20487452

## Overview:

Linear Regression with ElasticNet Regularization is implemented in this project using Python, NumPy for numerical computations, and Matplotlib for visualization. ElasticNet regularization aims to combine the strengths of L1 (Lasso) and L2 (Ridge) regularization approaches to provide a more balanced model. This strategy reduces overfitting while preserving model complexity and feature selection.

In this project, we manually created ElasticNet without using pre-built libraries like scikit-learn, as per project requirements. The model was trained, tested, and evaluated on real-world datasets, and visuals were supplied to help examine its performance.  
  
## Q1) What does the model you have implemented do and when should it be used?
  
### Ans :

In this project, we implemented Linear Regression with ElasticNet Regularization. The ElasticNet model is a combination of L1 (Lasso) and L2 (Ridge) regularization techniques, which helps in preventing overfitting and feature selection in regression problems. The model solves the linear regression problem by minimizing the loss function, while applying both types of regularization to the coefficients. The L1 regularization encourages sparsity, meaning it drives some coefficients to zero, which can be useful for feature selection. On the other hand, L2 regularization discourages large coefficients, helping to handle multicollinearity and preventing overfitting.
  
### ElasticNet is particularly useful when:

There are many correlated features in the dataset.
You want to perform both feature selection (via L1) and shrinkage (via L2).
You need a balance between the strengths of Lasso (feature selection) and Ridge (stability) regularization.
  
## Q2) How did you test your model to determine if it is working reasonably correctly?

### Ans :

To evaluate the correctness of our implementation, we employed k-fold cross-validation (5 folds). This method splits the dataset into multiple folds, where the model is trained on k-1 folds and tested on the remaining fold. This process repeats for each fold, ensuring that the model is trained and tested on different subsets of the data. Using cross-validation helps ensure that the model generalizes well and isn't overfitting to one particular train-test split. We measured the performance of the model using two key metrics :

•	Mean Squared Error (MSE): Indicates the average squared difference between actual and predicted values.

•	R-squared (R²): Measures how well the model explains the variance in the data.
  
We also visualized the results by plotting actual vs predicted values and residual histograms to assess model performance visually. The consistent R-squared values across different folds showed that the model was performing well.
  
##  Q3) What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)

#### Ans :

### We exposed several parameters to allow users to tune the performance of the ElasticNet model:

 • alpha: Controls the strength of the regularization. Higher values result in stronger regularization, making the model more conservative in selecting features.  
  
 • l1_ratio: Determines the mix between L1 (Lasso) and L2 (Ridge) regularization. A value closer to 1 emphasizes L1 (more feature selection), while a value closer to 0 emphasizes L2 (more coefficient shrinkage).  
  
 • max_iter: Defines the maximum number of iterations for the optimization process.  
  
 • tol: Sets the tolerance for determining when the optimization process has converged, providing control over the trade-off between accuracy and computation time.  
  
These parameters allow users to customize the regularization behavior and optimize the model for different datasets or use cases.
  
##  Q4) Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

### Ans :

Our implementation may face challenges with datasets that exhibit extreme non-linearity or contain a large number of outliers. Since ElasticNet is fundamentally a linear model, it may struggle to fit datasets with strong non-linear relationships. Additionally, sparse datasets or datasets with many missing values might not be handled effectively, as the model assumes complete and well-behaved data.  
 
### Given more time, we could work around these issues by:

• Introducing non-linear transformations or polynomial features to extend the model's ability to handle non-linear relationships.  
  
• Developing a preprocessing pipeline that handles missing values, outliers, and scaling issues before feeding the data into the model.  
  
However, some of these challenges are inherent to linear models, and addressing them fully might require using more advanced techniques, such as non-linear models (e.g., decision trees or neural networks) or robust outlier detection mechanisms.  
  
## Additional Libraries Used:- 

### For this project, we utilized the following Python libraries:

#### 1. Pandas:

Purpose: Pandas was utilized to handle and alter the dataset, allowing us to load, clean, and preprocess it quickly. It includes data structures like DataFrames, which make it simple to manage tabular data (such as the house price dataset).  
  
Why We Used It: Pandas is great for working with structured data and doing things like reading CSV files, handling missing values, and doing data transformations. Because our project involves regression on a real estate dataset, Pandas was required to prepare the data before feeding it into the model.  
  
#### 2. Matplotlib :

Matplotlib was used to generate visuals, such as scatter plots and histograms, to aid in the analysis of model performance.  
  
**Why Did We Use It?** :  Visualizing the model's outputs (e.g., actual versus predicted values, residual plots) allows us to assess how well the model matches the data. These visualizations also allow us to identify potential concerns, such as outliers or trends in the residuals, which provides further information about model correctness and opportunities for improvement.  
  
##  Code Structure and Implementation:

### 1. ElasticNet Class Implementation:

We created an ElasticNet class that contains the following important methods:  

fit(X, y): This approach trains the model based on the input data (X) and target labels (Y). The training method entails maximizing the coefficients with the regularization parameters.  
  
predict(X): Once the model has been trained, this method uses new input data (X) to forecast the target values.  
  
### 2. Cross Validation:  

We used k-fold cross-validation to evaluate the model's performance. This method divides the data into k parts, trains the model on k-1 parts, and then tests it on the remaining parts. This technique is done k times to ensure that each segment of the data is only utilized as a test set once. This method assesses how well the model generalizes to unseen data.  
  
## Code usage:

### Example of using the code:

To utilize the model, you will have to:  

1. Import the required libraries.  
2. Set up the ElasticNet model with specific hyperparameters.  
3. Train the model with the fit() technique.  
4. Use the predict() method to make predictions about outcomes.  
  
#### Below is a basic example of how to use the model :

#### Import the model
```python
from ElasticNet import ElasticNetModel
```
  
#### Initialize the ElasticNet model with custom parameters
```python
model = ElasticNetModel(alpha=0.5, l1_ratio=0.7, max_iter=1000, tol=1e-4)
```
  
#### Train the model with training data (X_train and y_train)
```python
model.fit(X_train, y_train)
```
  
#### Generate predictions on the test data
```python
predictions = model.predict(X_test)
```

#### Evaluate performance using Mean Squared Error (MSE)
```python
mse = np.mean((y_test - predictions) ** 2)
print(f'Mean Squared Error: {mse}')
```
  
###  Visualisation of Results:

In addition to training and prediction, we added visualizations to examine the model's performance:

This scatter plot compares the actual and predicted values. If the model is functioning properly, the points should closely follow the identity line (y = x).  
  
The Residual Histogram illustrates the distribution of residuals (the difference between real and expected values). A well-performing model should have residuals centered around zero and no discernible trend.  
  
These visualizations, created with Matplotlib, are useful for determining how well the model matches the data.  
