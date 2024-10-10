Group Members:
1. Spandana Vemula - A20527937
2. Hemanth Thathireddy - A20525346


# ElasticNet Regression from Scratch

## Project Overview
In this project, we have successfully developed an ElasticNet regression model from scratch using gradient descent. ElasticNet is a 
combination of L1 (Lasso) and L2 (Ridge) regularization. The goal was to minimize the cost function by using both types of regularization 
to prevent overfitting and handle multicollinearity.

We apply this model on the **auto-mpg** dataset, prezzozed from several car features to try to predict the MPG of a car from these features.


## Questions
# 1. What does the model you have implemented do and when should it be used?
Ans. Our implemented model was Elastic Net regression, an implementation that regularizes via both L1-norm (Lasso) and L2-norm regularization. 
It would come in handy for regression tasks when trying to predict a continuous target variable based on a set of features. 
ElasticNet would be very useful in the following cases:

We have to handle high-dimensional data-a large number of features-and want to perform feature selection, making less important features shrink.
There is multicollinearity present in the data, and both L1-sparsity and L2-smoothness regularizations are needed in order to avoid overfitting.
Application to Auto MPG Dataset: The following model has been applied to the auto-mpg dataset for the prediction of miles per gallon for different 
cars based on their engine displacement, horsepower, weight, and other feature parameters.

# 2. How did you test your model to determine if it is working reasonably correctly?
Ans. We then tested this ElasticNet model by training it on the full auto-mpg dataset and making predictions from its learned weights and 
bias. We then evaluated this model by computing the Mean Squared Error (MSE) on the predicted values. The MSE was computed as 12.2156, 
giving us a quantitative measure of how well the model did from a prediction accuracy perspective. Making appropriate plots to visualize 
the performance.
Actual vs Predicted Values Plot: The plot would visually compare how close the predicted MPG values had come to the actual values. 
Residuals vs Predicted Values Plot: Check to see if the errors are random; this would assure the model is not systematically overor 
under-predicting. Distribution of Residuals: Check if the errors follow a normal distribution. In doing all these steps, we could be 
certain that the model was working the way it should and the predictions made sense.

# 3. What parameters have you exposed to users of your implementation in order to tune performance?
Ans. The following parameters were exposed for tuning in our implementation of ElasticNet: 
alpha: Regularization strength - Higher values of alpha apply more regularization to help reduce overfitting.
l1_ratio: This regulates the ratio between L1/Lasso and L2/Ridge. A number closer to 1 will apply more L1 regularization, hence sparsity of features, while a number closer to 0 will apply more L2 regularization, hence smoothing the coefficients.
Learning_rate: Hyper-parameter that provides the step-size for the gradient descent optimization algorithm. The higher the learning rate, the faster convergence. The smaller the learning rate, the slower convergence, though it is more stable.
iterations: This regulates the number of steps or iterations for the gradient descent. The larger the number of iterations, 
the better the model converges, hence the lower the cost function.
These parameters will enable the user to vary the trade-off between bias vs. variance, allowing the recovery of the best 
performance of the model over different datasets.

# 4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental to the model?
Ans. The ElasticNet model may, in its present form, face challenges with:

Non-linear relationships: It assumes the relationship between the features and the target variable is linear. For non-linear relationships, this model may not work that well. We might want to apply feature engineering or non-linear transformations so that we can work our way around this issue.
Outliers: Extreme outliers in the dataset may disproportionately affect the performance of the model. L1 regularization helps to some extent, but more sophisticated techniques of outlier detection can be applied beforehand.
Feature scaling: ElasticNet does expect features to be of the same scale, and that is precisely why we applied standardization-scalings as a pre-processing step. If not done, the model may fail or not work well.
If time permits, we could go through some of the more advanced techniques such as adding non-linear terms and kernel methods to handle non-linearity, or doing robust scaling to take care of outliers more effectively.


### What we have done:
- Implemented ElasticNet regression from scratch using gradient descent.
- Cleaned **auto-mpg** dataset: missing value handling, unnecessary column removal.
- ElasticNet model will be trained on the full dataset and evaluated using Mean Squared Error.
- Visualized different model performance and characteristics of the dataset using various plots.

## Files in the Project
- **Spandana.ipynb**: This notebook contains all code for:
- Loading and preprocessing of the data.
- ElasticNet model - implementation and training.
- Prediction and calculation of MSE.
- Visualizations of Model Performance and Relationships in Features.

## Dataset
The **auto-mpg** dataset was used for this project. The dataset contains the following features:
- `mpg`: Miles per gallon (target variable).
- `cylinders`: Number of cylinders in the engine.
- `displacement`: Engine displacement (cubic inches).
- `horsepower`: Engine horsepower.
- `weight`: Vehicle weight.
- `acceleration`: Time to accelerate from 0 to 60 mph (seconds).
- `model year`: Year of manufacture.

### Preprocessing Steps
- Removed the columns for `car name` and `origin`, since these columns were not relevant in predicting the output.
- The `horsepower` column was converted to numeric, thus missing values were dropped as rows containing `NaN`.
- Separate the dataset into features (`X`) and the target variable (`y`).

## ElasticNet Implementation
We implemented ElasticNet from scratch using gradient descent. The cost function minimized was:

\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 + \alpha \left( \frac{(1 - \rho)}{2} \sum_{j=1}^{n} \theta_j^2 + \rho \sum_{j=1}^{n} |\theta_j| \right)
\]

Where:
- \( h_\theta(x) \) is the hypothesis function.
- \( \alpha \) is the regularization strength.
- \( \rho \) is the L1 ratio, controlling the balance between L1 and L2 regularization.

### Gradient Descent
Model parameters, which include weights and bias, were optimized using Gradient Descent with an iterative update rule such that the cost function became minimized. Since the model took more than 1000 iterations to learn, we monitored cost after regular intervals to make sure that convergence happened.

### Final Model
The final model parameters (weights and bias) were learned, and we evaluated the model using Mean Squared Error (MSE).

## Visualizations
We generated various visualizations to understand the model performance and the dataset:

1. **Actual vs Predicted Values Plot**: 
   - Shows how close the model’s predictions are to the actual values. Ideally, the points should lie close to the diagonal line.
   
2. **Residuals vs Predicted Values Plot**: 
   - Helps us visualize the errors (residuals) and check if there are any patterns. Randomly scattered residuals indicate a well-fitted model.
   
3. **Distribution of Residuals (Histogram + KDE)**: 
   - Displays whether the residuals follow a normal distribution, indicating that the model's errors are unbiased.
   
4. **Feature Correlation Matrix (Heatmap)**: 
   - Visualizes the correlation between different features in the dataset. High correlations can indicate multicollinearity.
   
5. **Learning Curve (Training Size vs MSE)**: 
   - Shows how the model's performance improves as we increase the size of the training dataset. This helps in understanding whether the model is underfitting or overfitting.
   
6. **Predicted vs Actual Values Histogram**: 
   - Compares the distribution of predicted values and actual values, helping us understand if the model is systematically overestimating or underestimating the target variable.
   
7. **Coefficients Visualization**: 
   - Visualizes the magnitude and direction of the coefficients learned by the model. Larger coefficients indicate more influence on the target variable.

8. **MSE over Iterations (Convergence Plot)**: 
   - Plots the MSE over the iterations of gradient descent to observe the model's learning process and convergence.

## Model Evaluation
The model was evaluated using Mean Squared Error (MSE) on the **auto-mpg** dataset. The final MSE was:

\[
\text{Mean Squared Error (MSE)} = 12.2156
\]

This value indicates how well the model predicted the miles per gallon (MPG) based on the features of the cars.

## Conclusion
We successfully implemented ElasticNet regression from scratch, successfully trained on the **auto-mpg** dataset, and successfully evaluated its performance by using MSE. We also drew several visualizations that gave insight into how well our model performed, as well as how the different features interacted with one another.

Feel free to explore the **Spandana.ipynb** notebook to see the full implementation and visualizations.
