
# Linear Regression with Elastic Net Regularization
Group Members: 
Vishwas Reddy Dodle (A20562449)
Yaswanth Mopada (A20585424)
-----------------------------------------------------------------------------------------------
# Project Overview
This project implements a linear regression model with Elastic Net regularization, which is a combination of L1 (Lasso) and L2 (Ridge) penalties. Elastic Net is particularly useful in scenarios where features are correlated or when feature selection is required.
-----------------------------------------------------------------------------------------------
# Usage Instructions
## To use the model:

I: Initialize the ElasticNetLinearRegression class with desired parameters.
II: Call the fit() method on your training data.
III: Use the predict() method to make predictions on new data.
IV: Evaluate the model using the mse() method for Mean Squared Error calculation.
-----------------------------------------------------------------------------------------------
# 1. What does the model you have implemented do and when should it be used?
This model performs linear regression with Elastic Net regularization, which is a combination of L1 (Lasso) and L2 (Ridge) regularization. It is designed to handle both overfitting and multicollinearity in datasets, where it penalizes large coefficients (L2) and encourages sparsity (L1). It is useful when features are correlated or when feature selection is necessary in the model.

# 2. How did you test your model to determine if it is working reasonably correctly?
I tested the model using a small synthetic dataset where the linear relationship is known. I computed predictions and checked that the Mean Squared Error (MSE) decreased as the model optimized the parameters during gradient descent. The model was also checked for handling both Lasso (L1) and Ridge (L2) penalties by varying the l1_ratio parameter.

# 3. What parameters have you exposed to users of your implementation in order to tune performance?
alpha: Controls the strength of regularization (penalty).
l1_ratio: Determines the balance between L1 and L2 regularization.
learning_rate: Adjusts the step size for gradient descent optimization.
iterations: Sets the number of iterations for gradient descent to converge.
These parameters can be tuned based on the dataset and model requirements to balance bias and variance.

# 4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these, or is it fundamental to the model?
The implementation may struggle with:
Very large datasets due to the basic gradient descent approach. A more efficient optimization algorithm like stochastic gradient descent or coordinate descent could improve scalability.
Extreme multicollinearity: While Elastic Net helps with multicollinearity, very high correlations could still pose challenges. Further testing could involve adding features for automatic feature selection.
-----------------------------------------------------------------------------------------------------