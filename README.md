Student Name and A#

Harlee Ramos	 A20528450

Andres Orozco A20528634

**README FILE**

This document contains two parts. The information on how to run the program and the answers to the Project 1 questions. 

How to run the ElasticNet model:

Instructions on how to use the ElasticNet Regularization Model: The `ElasticNetModel` is a custom implementation of ElasticNet regression, combining L1 (Lasso) and L2 (Ridge) regularization with gradient descent optimization. It’s suitable for datasets with high dimensionality and multicollinearity. 

Importing the Model

In your Python environment, import the `ElasticNetModel following the code and examples below:

from elasticnet.models.ElasticNet import ElasticNetModel

Training the Model with fit
To train the model, use the fit method. It requires:
X: A 2D NumPy array with shape [n_samples, n_features] representing the features.
y: A 1D NumPy array with shape [n_samples] representing the target values.
Example:

import numpy as np
from elasticnet.models.ElasticNet import ElasticNetModel

# Generate example data
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = 2 * X[:, 0] - 3 * X[:, 1] + 1.5 * X[:, 2] + np.random.normal(0, 0.1, 100)

# Initialize the model
model = ElasticNetModel(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-5, learning_rate=0.01)

# Fit the model
results = model.fit(X, y)
Making Predictions with predict
To predict with the fitted model, use the predict method:
X: A 2D NumPy array with the same number of features as the training data.
Example:
# Predict using the trained model
y_pred = results.predict(X)
import numpy as np
from elasticnet.models.ElasticNet import ElasticNetModel

# Generate sample data
X = np.random.rand(100, 3)
y = 4 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 0.1, 100)

# Initialize and fit the model
model = ElasticNetModel(alpha=0.05, l1_ratio=0.7, max_iter=1000, tol=1e-5, learning_rate=0.01)
results = model.fit(X, y)

# Make predictions
y_pred = results.predict(X)

print("Sample predictions:", y_pred[:5])
Notes
Ensure that X and y are formatted as NumPy arrays before using fit.
The predict method can be used for any dataset with the same number of features as the training data.
Answer to the Project 1 questions:

**Brief introduction**

The machine learning project 1 implements an ElasticNet regression model, a linear regression technique that incorporates L1 (Lasso) and L2 (Ridge) regularization penalties. ElasticNet addresses the limitations of Lasso and Ridge by combining them, making it suitable for high-dimensional data with potentially collinear features.
The model is optimized using gradient descent, an iterative method for minimizing functions by adjusting coefficients in the direction of the steepest descent of the gradient. ElasticNet’s objective function can be written as:

The Elastic Net regression aims to minimize the following objective function:

<p>arg min<sub>&beta;</sub> (1 / 2n) * ||y - X&beta;||<sub>2</sub><sup>2</sup> + &alpha; * (1 - l1_ratio / 2) * ||&beta;||<sub>2</sub><sup>2</sup> + l1_ratio * ||&beta;||<sub>1</sub></p>

Where
 represents the estimated coefficients (or parameters) of the ElasticNet regression model that minimize the objective function.


arg min this denotes the operation of finding the value of  (the coefficient vector) that minimizes the objective function.
12n​∣∣y-Xβ∣∣22  is the residual sum of squares (RSS), which measures how well the model fits the data.
y is the vector of observed target values.
Xβ are the predicted values (based on the feature matrix X and coefficients  .
∣∣y-Xβ∣∣22 is the squared Euclidean distance (L2 Norm) between the true target y and the predicted values Xβ, which represents the total error.
12n this part uses a 2n factor for scaling and simplifies the gradient calculation, n represents the number of observations/data points.
 is the regularization strength. It balances the trade-off between minimizing the error (RSS) and shrinking the coefficients to reduce model complexity.
1-l1_ratio2​∣∣β∣∣22 is the Ridge penalty to control overfitting.
∣∣β∣∣22 The L2 norm (sum of squares) of the coefficients  which penalizes large values of  to prevent overfitting.
1-l1_ratio2  This controls the contribution of the L2 penalty, based on the L1 ratio parameter.


l1_ratio⋅∣∣β∣∣1 is the Lasso penalty induce sparsity.
⋅∣∣β∣∣1 The L1 norm (sum of absolute values) of the coefficients \beta, which encourages sparsity by shrinking some coefficients to zero.
l1_ratio This controls the contribution of the L1 penalty in the regularization mix. A value of 1 means only Lasso, while a value of 0 means only Ridge.

Soft Thresholding and proximal operator perspective
=prox1argmin12n​∣∣y-Xβ∣∣22​+22​∣∣β∣∣22​

1. What does the model you have implemented do and when should it be used?
The ElasticNet regression model implemented in this project is a linear regression technique that combines L1 (Lasso) and L2 (Ridge) regularization penalties. This combination allows the model to address the limitations of using either Lasso or Ridge alone, such as over-penalization (Lasso) or failing to select relevant features (Ridge). The optimization process uses gradient descent, which iteratively adjusts the coefficients by minimizing the objective function in the direction of the steepest descent. This method allows the model to balance between fitting the training data and controlling complexity through regularization.
The ElasticNet model is particularly useful when dealing with datasets that have:
- **High Dimensionality**: It performs well when the number of features is greater than the number of observations or when many features are irrelevant. The L1 penalty helps in selecting the most relevant features by shrinking some coefficients to zero, making the model easier to interpret.
- **Multicollinearity**: ElasticNet is effective when features are highly correlated, as it combines the L2 penalty to stabilize the solution and avoid overfitting that could arise from collinear predictors. This ensures a more balanced coefficient estimation compared to Lasso alone.
- **Feature Selection and Regularization**: ElasticNet can select a subset of features while still applying a penalty to the magnitude of other coefficients, which makes it a versatile option for improving model interpretability.
The model should **not be used** when:
- **Data is scarce**: If the dataset has very few observations, simpler models like Ridge or Lasso alone might be more appropriate.
- **No multicollinearity exists**: In cases where features are not correlated, a simpler model might suffice.
- **Real-time predictions are required**: Since this implementation uses gradient descent, it may not be as fast as some optimized libraries (e.g., scikit-learn) for large-scale data.
Overall, this implementation is a great fit for scenarios where feature selection, handling multicollinearity, and scalability across a range of regularization strengths are crucial.
2. How did you test your model to determine if it is working reasonably correctly?

We tested the custom ElasticNet model by comparing it against the scikit-learn ElasticNet implementation across six progressively complex datasets. These tests included simple linear data with a single feature, linear data with multiple features, nonlinear data, collinear data, periodic data, and high-dimensional data. Each test allowed us to observe how the model performed in different scenarios, such as handling nonlinearity, multicollinearity, and complex feature interactions. We evaluated model performance using accuracy metrics like R² (coefficient of determination) and MSE (mean squared error) to ensure the model's predictive accuracy. By comparing the results with scikit-learn's implementation, we validated the custom model's performance and identified its strengths and weaknesses in handling diverse types of data. The ElasticNet testing is part of the deliveries of this project and can be found in the test section of the GitHub environment. 

3. What parameters have you exposed to users of your implementation to tune Performance?

The ElasticNet model provides several adjustable parameters to help users fine-tune performance based on their data and desired outcomes:

- **Alpha (α)**: Controls the overall strength of regularization. Higher values of alpha increase regularization, which helps to prevent overfitting but can lead to underfitting if set too high. Users can adjust this to find a balance between fitting the data closely and keeping the model simple.

- **L1 Ratio (L1/L2 Ratio)**: Determines the balance between L1 (Lasso) and L2 (Ridge) penalties. A value of 1.0 means pure Lasso (more feature selection), while 0.0 means pure Ridge (more smoothing). Users can adjust this to control the level of sparsity and feature selection in their model.

- **Maximum Iterations (max_iter)**: Sets the maximum number of iterations for the optimization process. If the model struggles to converge, increasing this value can help. A lower value can be used for faster training when precision is less critical.

- **Tolerance (tol)**: Defines the stopping criteria for the optimization. Lower tolerance values make the model run longer for more precise results, while higher values can speed up training when quick results are needed.

- **Learning Rate**: Adjusts the step size of the gradient descent during each iteration. A smaller learning rate provides more precise optimization but can be slower, while a larger learning rate speeds up training but may risk overshooting the optimal solution.

- **Model Coefficients (coef_) and Intercept (intercept_)**: After training, these attributes represent the model's learned parameters. While users don’t adjust them directly, they can inspect these to understand how the model has adapted to their data.

Users can tweak these parameters to improve the model's performance until they find the best fit for their data, making the ElasticNet model adaptable to various datasets and scenarios.

4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental to the model?

The custom ElasticNet model has been tested extensively and handles various types of data effectively, including linear, nonlinear, and collinear datasets. However, there are some considerations that could be improved with further refinement:

- **Handling Large Datasets**: While the model now processes larger datasets more effectively, the gradient descent approach remains less efficient than the coordinate descent method used by libraries like scikit-learn. This is due to gradient descent’s need for multiple iterations across the dataset, which can become time-consuming as data size increases. Given more time, adapting the model to use mini-batch gradient descent or stochastic gradient descent (SGD) could further enhance its scalability.

- **Multicollinearity**: The model performs well with collinear features, but certain scenarios with extreme collinearity can still pose challenges. Although regularization helps, the gradient descent-based optimization might not adjust as efficiently as scikit-learn's built-in methods. Improving this could involve incorporating pre-processing steps like Principal Component Analysis (PCA) to reduce feature correlations before training.

- **Optimization Efficiency**: The current gradient descent implementation requires careful tuning of hyperparameters like learning rate and tolerance to achieve optimal convergence. A more advanced approach, such as using adaptive learning rates or switching to coordinate descent, could provide better convergence properties without the need for extensive tuning.

- **Handling Missing Data and Categorical Variables**: Like many custom implementations, this model assumes clean, numerical input data. Handling missing values or categorical features requires pre-processing before training. While these issues are not fundamental to the model's design, adding imputation methods or encoding techniques could make the model more adaptable to real-world datasets with these characteristics.

In summary, while the current implementation is robust for many use cases, it could be further optimized for large-scale datasets and enhanced with pre-processing techniques for more complex data. These improvements are achievable with time and are not fundamental limitations of the ElasticNet model itself, but rather, opportunities for refining its performance.

