# ElasticNet Regression

## Table of Contents
- Project Overview
- Key Features
- Setup and Installation
- Model Explanation
- How to Use
- Code Examples
- Adjustable Parameters
- Known Limitations
- Contributors
- Q&A

## Project Overview
This project provides a fully custom implementation of ElasticNet Regression, built from the ground up using only NumPy and pandas. No external machine learning libraries like scikit-learn or TensorFlow have been used. This implementation aims to provide a clear understanding of ElasticNet's operation and shows how the model may be optimized via gradient descent.

Combining L1 (Lasso) and L2 (Ridge) regularization, ElasticNet is a linear regression model that works well for tasks involving correlated features or feature selection. Gradient descent is utilized for model optimization.

## Key Features
- **Custom ElasticNet Regression**: Implements both L1 (Lasso) and L2 (Ridge) regularization for linear regression.
- **Gradient Descent Optimization**: Manually optimizes weights using gradient descent, allowing full control over the learning process.

## Setup and Installation
### Prerequisites
- Python 3.x
- NumPy
- pandas

### Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/priyanshpsalian/ML_Project1.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Model Explanation
Combining the benefits of L1 and L2 regularization, ElasticNet is a regularized version of linear regression. It functions effectively in cases when we want both variable selection (L1) and coefficient shrinkage (L2), or when features are coupled.

### Objective Function

The objective of ElasticNet is to minimize the following:


- **Alpha** controls the strength of regularization.
- **l1_ratio (Lambda)** determines the mix between L1 (Lasso) and L2 (Ridge).

## How to Use
You can initialize and train the ElasticNet model using the provided `ElasticNetModel` class:
```python
from elasticnet import ElasticNetModel

# Initialize the model
model = ElasticNetModel(alpha=1.0, l1_ratio=0.5, max_iter=2000, convergence_criteria=1e-4, step_size=0.005, bias_term=True)

# Fit the model to data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

```
## Code Examples
This is just an example code:
```python
# Fit the model
outcome = model.fit(X_train, y_train)

# Predict
y_pred = outcome.predict(X_test)

# Evaluate
r2 = outcome.r2_score(y_test, y_pred)
rmse = outcome.rmse(y_test, y_pred)

print(f"R² Score: {r2}")
print(f"RMSE: {rmse}")
```

## Adjustable Parameters

- **self**: The instance of the ElasticNetModel class that this method is called from - used to access class properties and other methods.
- **alpha**: Overall strength of regularization. Must be a positive float. Its value is 1 by default.
- **l1_ratio**: This parameter balances the L1 (Lasso) and L2 (Ridge) penalties, where 0 indicates pure Ridge and 1 indicates pure Lasso. The default setting is 0.5
- **max_iter**: The maximum number of passes over the training data. It defines the number of iterations in gradient descent optimization. Higher values allow more fine-tuning, at the cost of more computation. The default is 2000
- **convergence_criteria**: Tolerance for stopping criteria. If the difference between iterations is less than this value, then the training stops. The default is 1e-4
- **step_size**: Step size determines the amount that coefficients are altered during each step of gradient descent. Small values can lead to slower convergence but more precise results. Default = 0.005
- **bias_term**: Boolean indicating if an intercept should be added to the model. If True, then an intercept term will be added. Default = True


## Known Limitations
Increasing Decline Convergence: In situations with significant multicollinearity or on huge datasets, the model may converge slowly. Convergence may be enhanced by alternative optimization methods like coordinate descent.

Precision: Compared to closed-form solutions, gradient descent may not be able to reach the level of precision needed for some applications.


## Contributors
- Priyansh Salian (A20585026 psalian1@hawk.iit.edu)
- Shruti Ramchandra Patil (A20564354 spatil80@hawk.iit.edu)
- Pavitra Sai Vegiraju (A20525304 pvegiraju@hawk.iit.edu)
- Mithila Reddy (A20542879 Msingireddy@hawk.iit.edu)

## Q&A

### What does the model you have implemented do, and when should it be used?
ElasticNet Regression is designed to handle regression tasks involving multicollinearity (correlation between predictors) and feature selection. It combines the L1 (Lasso) and L2 (Ridge) penalties to strike a compromise between coefficient shrinkage and variable selection.

### How did you test your model to determine if it is working reasonably correctly?
The model was tested using synthetic data that had established correlations between predictors and target variables. R² and RMSE measures were used to assess the accuracy of the model by comparing predictions to actual values.

### What parameters have you exposed to users of your implementation in order to tune performance?
- **Alpha**: The strength of regularization, which manages both L1 and L2 penalties.
- **l1_ratio**: The proportion of L2 (Ridge) penalties to L1 (Lasso) penalties.
- **step_size**: The gradient descent step size.
- **max_iter**: The quantity of iterations used to optimize gradient descent.
- **convergence_criteria**: Tolerance for the halting criteria. If the progress between iterations is less than tol, the training process will end. Default is 1e-4.
- **bias_term**: Boolean indicating whether an intercept should be fitted. If True, an intercept term is introduced into the model. Default is True.

### Are there specific inputs that your implementation has trouble with? Given more time, could you work around these, or is it fundamental to the model?
Large datasets or datasets with extreme multicollinearity may provide challenges for the existing solution since gradient descent may converge slowly. Coordinate descent is one optimization technique that could be used to accelerate convergence if given extra time, particularly in high-dimensional environments.




















