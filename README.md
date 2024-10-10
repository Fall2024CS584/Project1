
# ElasticNet Linear Regression Model

## Authors
- **Sankar Ganesh Paramasivam** - A20553053
- **Neelarapu Tejaswini** - A20592053
- **Vijaya Sai Dasari** - A20540356
- **Aravinth Ananth** - A20537468

## Overview
This project implements a **Linear Regression model** with **ElasticNet regularization**, which combines L1 (Lasso) and L2 (Ridge) regularization techniques. ElasticNet helps balance the sparsity of Lasso and the stability of Ridge by adjusting the `l1_ratio` parameter. This model is particularly useful for datasets with many correlated features, offering both feature selection and improved generalization.

### Key Features
- **L1 regularization** (Lasso) encourages sparsity in the coefficients, leading to feature selection.
- **L2 regularization** (Ridge) ensures stability by shrinking the coefficients and helps avoid overfitting.
- **ElasticNet** combines both L1 and L2 regularization, allowing for a balance between sparsity and stability.

---



### Q1: What does the model do, and when should it be used?

The model is a **Linear Regression model with ElasticNet regularization**, which is beneficial when:
- Handling multicollinearity (correlated features).
- Performing feature selection by shrinking some coefficients to zero.
- Preventing overfitting by controlling the complexity of the model.

It is particularly useful for datasets with many predictors or highly correlated features, providing a balance between sparsity (via Lasso) and stability (via Ridge).

### Q2: How did you test the model?

We tested the model using the following metrics:
- **R-squared (R²):** Measures the proportion of variance in the dependent variable explained by the independent variables.
- **Mean Absolute Error (MAE):** Provides the average magnitude of prediction errors.
- **Root Mean Squared Error (RMSE):** Gives higher weight to larger errors, highlighting significant deviations.

These metrics were applied to test datasets to ensure reasonable predictions and assess the model's ability to generalize without overfitting or underfitting.

"In addition, we implemented a KDE plot to compare the distribution of actual values (**y_test**) and predicted values (**final_predictions**). This plot shows how closely the predicted values align with the actual data. The closer the two curves are, the better the model's performance."

### Q3: What parameters can users tune to improve performance?

Users can tune the following key parameters in the ElasticNet model:
- **Alpha (`alpha`):** Controls the strength of regularization. Higher values increase regularization.
- **L1 Ratio (`l1_ratio`):** Balances L1 (Lasso) and L2 (Ridge). Values range from 0 (pure Ridge) to 1 (pure Lasso).
- **Max Iterations (`max_iter`):** Sets the maximum number of iterations for convergence.
- **Tolerance (`tol`):** Defines the threshold for stopping the optimization process.

You can use **Grid search** to automatically test different combinations of these parameters to optimize the model based on metrics such as R² or RMSE.

### Q4: What specific inputs can the model struggle with?

The model may struggle with:
- **Imbalanced datasets:** The model can be biased towards dominant features/classes due to linear regression’s nature.
- **Extreme outliers:** Outliers can disproportionately influence coefficients, especially if L2 regularization isn’t strong enough.

Potential solutions:
- Implement **robust scaling** to reduce the effect of outliers.
- Apply **resampling techniques** to address imbalanced datasets or adjust the regularization based on data distribution.

---

## File Structure

- `test_ElasticNetModel.py`: Demonstrates the usage of the ElasticNet model, including loading the dataset, initializing the model, training, and testing.
- `ElasticNet.py`: Contains the custom implementation of the ElasticNet model with hyperparameter options (`alpha`, `l1_ratio`, `max_iter`, `tol`).
- `gridsearch.py`: Implements a grid search for hyperparameter tuning, optimizing the ElasticNet model via cross-validation.
- `checker.py`: Utility functions for evaluating and validating the model, including performance metric calculations.

---

## Code Usage

## Installation

To install the required dependencies, follow these steps:

1. Make sure you have Python installed.
2. Navigate to the project directory.
3. Install the required packages by running the following command:

```bash
pip install -r requirements.txt
```
### Running the Model

To run the ElasticNet model, use the following command:

```bash
python -m elasticnet.tests.test_ElasticNetModel
```

### Import the Model

```python
from ElasticNet import ElasticNetModel
```

### Initialize the ElasticNet Model

```python
model = ElasticNetModel(alpha=0.5, l1_ratio=0.7, max_iter=1000, tol=1e-4)
```

### Train the Model

```python
model.fit(X_train, y_train)
```

### Generate Predictions

```python
predictions = model.predict(X_test)
```

### Evaluate the Model

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
```

---

## Grid Search for Hyperparameter Tuning

To optimize the model’s hyperparameters using **Grid Search**:

```python
from gridsearch import GridSearch

param_grid = {
    'alpha': [0.1, 0.5, 1.0],
    'l1_ratio': [0.3, 0.7, 1.0]
}

# Perform grid search
grid_search = GridSearch(model, param_grid)
best_params = grid_search.fit(X_train, y_train)
print(f"Best parameters: {best_params}")
```

This will help find the optimal hyperparameters for your dataset, enhancing performance and accuracy.


