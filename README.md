# Project 1 

# ElasticNet Model Implementation

## What does the model you have implemented do and when should it be used?

The ElasticNet model is a regularized linear regression model that combines both L1 (Lasso) and L2 (Ridge) penalties. It is useful in scenarios where the dataset has many features, and you want to both reduce the number of features (via L1 regularization) and prevent overfitting (via L2 regularization). This model is especially effective when dealing with datasets that may have multicollinearity or when feature selection is needed.

The model can be used when:
- There is a need to balance feature selection (Lasso) with the ability to include small, non-zero coefficients (Ridge).
- The dataset may have high-dimensionality and you want to avoid overfitting while also regularizing the coefficients.
- The user seeks a regularized linear model that offers a compromise between Lasso and Ridge regression.

## How did you test your model to determine if it is working reasonably correctly?

To verify the correctness of the model, the following tests were performed:
- **Unit tests with pytest**: We wrote unit tests for both the `fit` and `predict` methods. The tests check if the model's coefficients and intercept are properly updated during training, and that predictions on test data are reasonably close to expected values.
- **Model Evaluation**: We evaluated the model on a synthetic dataset generated using known coefficients. After fitting the model to the data, we measured its performance using Mean Squared Error (MSE) and R-squared (R²) metrics. The model produced an MSE of `0.0147` and an R² of `0.9993`, indicating excellent fit and performance on the test data.
- **Visualization**: We also visualized the model's predictions against the actual test data using a scatter plot to confirm the model's accuracy visually.

## What parameters have you exposed to users of your implementation in order to tune performance?

The following parameters are exposed to the users to tune the performance of the ElasticNet model:
- `alpha`: The regularization strength, controlling the overall amount of shrinkage applied to the coefficients. A higher value will lead to more regularization.
- `l1_ratio`: Controls the mix between L1 (Lasso) and L2 (Ridge) penalties. A value of `1.0` corresponds to pure Lasso, and `0.0` corresponds to pure Ridge. Values in between balance the two.
- `max_iter`: The maximum number of iterations for the coordinate descent algorithm to converge.
- `tol`: Tolerance for stopping criteria. The algorithm stops when the coefficients change less than this value.
- `epochs`: Number of epochs to train the model, where an epoch is a complete pass through the entire dataset.

### Basic Usage Example:

```python
# Example usage of ElasticNetModel

# Import Elastic Net Model
from elasticnet.models.ElasticNet import ElasticNetModel

# Initialize the ElasticNet model with chosen hyperparameters
model = ElasticNetModel(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-4, epochs=10)

# Fit the model on training data
results = model.fit(X_train, y_train)

# Make predictions on test data
y_pred = results.predict(X_test)

# Evaluate the model
mse, r2 = results.evaluate(X_test, y_test)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```

## Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

The model may face difficulties with:

- `Non-linear relationships`: Since this is a linear model, it is not suited for datasets where the relationship between the independent variables and the target is non-linear. Non-linear transformations or switching to a non-linear model may help address this.

-`Highly correlated features`: While ElasticNet handles multicollinearity better than Lasso, very high correlation between features could still lead to challenges. Dimensionality reduction techniques such as PCA could be used prior to modeling to mitigate this issue.

- `Very small datasets`: The model's performance may suffer when there is very limited data, as regularization can dominate and lead to underfitting. In such cases, tuning the alpha and l1_ratio values or using a simpler model might work better.

Given more time, some improvements could include:
-`Adaptive learning rates or gradient descent based optimization to improve convergence speed and accuracy.`
-`Adding cross-validation to automatically select optimal hyperparameters (alpha, l1_ratio) based on the dataset.`
