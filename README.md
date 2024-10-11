# ElasticNetModel - Linear Regression with ElasticNet Regularization

## Team Members

- **Darshan Sasidharan Nair** : dsasidharannair@hawk.iit.edu
- **Ishaan Goel** : igoel@hawk.iit.edu
- **Ayesha** : asaif@hawk.iit.edu
- **Ramya** : rarumugam@hawk.iit.edu

## 1. Overview

The `ElasticNetModel` class implements **Linear Regression** with **ElasticNet Regularization**, which is a combination of **L1 (Lasso)** and **L2 (Ridge)** regularization techniques. This model is used for:

- **Linear regression tasks** where you want to predict a continuous target variable.
- **Feature selection** when you have a large set of features and want to eliminate irrelevant ones (due to L1 regularization).
- **Dealing with multicollinearity** in the dataset, as L2 regularization stabilizes the model and reduces variance.

### When to Use:

- ElasticNet should be used when the data has high-dimensional features and some of the features are expected to be irrelevant.
- It’s useful when there is **multicollinearity** (i.e., when predictors are correlated) in the data.
- Use it if you want to **select important features** and **reduce model complexity**.

## 2. Testing and Validation

To test if the model is works well, we performed the following tests:

- **Initial Test**:
  The model was first tested with synthetic datasets where the relationship between input features and target values was known. The model correctly identified trends and produced reasonable coefficient estimates.
- **Real Data**:
  The model was also validated on real-world datasets such as the Adverstising dataset, where it demonstrated expected behavior by regularizing feature weights, selecting important features, and generating sensible predictions.
- **Benchmarking**:
  The model was compared to `sklearn`'s `ElasticNet` implementation for consistency in predictions.

### Metrics for Evaluation:

- **Mean Squared Error (MSE)**: used to check how well the predicted values align with the actual target values.
- **Coefficient Magnitude**: observing how regularization influences the weight values, especially when tuning the `lambda1` parameter.
- **Real VS True Graph**: To check how close the predicted values were to the actual values and if the model actually fits the data points
- **Residual Graph/ Histogram**: To check the distribution of the residuals and enure homoscedasticity

### Correctness was established through:

- **Convergence** of the loss function.
- **Stability** in coefficients when tuning `lambda1`.
- Comparing predictions against a baseline linear regression model.

## 3. Parameters for Tuning

The model can be optimized by tuning the following hyper-parameters:

- **lambda1 (default: 0.5)**: This is the L1 regularization parameter (used in ElasticNet to control the penalty for the magnitude of the coefficients). Larger values encourage sparser solutions, where some feature coefficients may become zero. A value closer to 0 leans more towards L2 regularization, making it similar to Ridge regression.
- **threshold (default: 0.000001)**: This is the convergence threshold for gradient descent. The training process stops when the change in model weights (beta coefficients) is less than this threshold, indicating that the model has converged. Smaller values might lead to more accurate solutions but could require more iterations.
- **learning_rate (default: 0.000001)**: This is the step size for each iteration of gradient descent. It controls how much the model updates its weights after each step. A smaller learning rate ensures more stable convergence but might require more iterations to reach an optimal solution.
- **scale (default: False)**: This is a Boolean flag indicating whether to scale the features using `MinMaxScaler`. When set to `True`, the input features are scaled to a specified range, defined by `scale_range`.
- **scale_range (default: (-10, 10))**: This is the range within which the features are scaled when scale is set to `True`. This ensures the input data is transformed into a desired range, which can be helpful for algorithms that are sensitive to the magnitude of features.

## 4. Challenges and Limitations

### Potential Issues:

- **Highly Correlated Features**: While `ElasticNetModel` handles multicollinearity better than standard linear regression, data with significantly high correlations between features might still cause some instability in weight updates.
- **Nonlinear Relationships**: The model assumes a linear relationship between features and the target. If there is a nonlinear relationship in the data, `ElasticNetModel` will not perform well unless features are transformed appropriately prior to inputting them into the model. It also performs poorly on binary or general categorical data.
- **Imbalanced Datasets**: If the dataset has a high class imbalance, `ElasticNetModel` might struggle to fit the data well.
- **Large Values**: Datasets with large values need to be scaled down otherwise the runtime can grow very quickly.

### Improvements:

Given more time, improvements could include:

- Implementing **cross-validation** to automatically determine optimal values for `lambda1`.
- Developing **early stopping criteria** during gradient descent to avoid unnecessary iterations once the loss function stabilizes.
- Implementing strategies to **handle missing data** would make the model more robust for real-world datasets, which often contain incomplete data.

---

### Usage Example:

```python
# Initialize the model
elastic_net = ElasticNetModel(lambda1=0.7, learning_rate=0.01, threshold=0.0001, scale=True)

# Fit the model to training data
elastic_net.fit(X_train, y_train)

# Predict using test data
y_pred = elastic_net.predict(X_test)
```

A working usage example can be seen in the test_ElasticNetModel.py file. This can used for reference for future testing.

The datasets provided can also be used to ensure that the model works. This is few of the real world datasets the team used to check the validity of the model.

This implementation offers flexibility to users to experiment with various parameter settings, providing both L1 and L2 regularization, making it suitable for various types of linear regression problems.
