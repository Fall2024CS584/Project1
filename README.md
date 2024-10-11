
# Project 1 

Project Members:

1.Satwik Sinha
A20547790
ssinha20@hawk.iit.edu

2.Aditya Ramchandra Kutre  
CWID : #A20544809
akutre@hawk.iit.edu

3.Tejaswi Yerra
CWID : #A20545536
tyerra@hawk.iit.edu

# ElasticNet Linear Regression Implementation

## Overview

This project implements **Linear Regression with ElasticNet Regularization** from first principles. ElasticNet combines both L1 (Lasso) and L2 (Ridge) regularization to enhance model performance, especially in scenarios with high-dimensional data or multicollinearity among features.

## **What does the model you have implemented do and when should it be used?**

The implemented **ElasticNet** model performs linear regression while applying a combination of L1 and L2 penalties to the loss function. This approach offers several advantages:

- **Feature Selection:** L1 regularization encourages sparsity, effectively selecting relevant features.
- **Handling Multicollinearity:** L2 regularization mitigates issues arising from highly correlated predictors.
- **Improving Generalization:** The combined regularization prevents overfitting, enhancing the model’s ability to generalize to unseen data.

**When to use ElasticNet:**

- When dealing with datasets that have a large number of predictors.
- When there is multicollinearity among features.
- When feature selection is desired alongside regression.
- When seeking a balance between L1 and L2 regularization benefits.

## **How did you test your model to determine if it is working reasonably correctly?**

Testing was conducted through the following approaches:

- **Synthetic Data Generation:** Utilized the provided `generate_regression_data.py` script to create synthetic datasets with known coefficients and noise levels, validating the model's ability to recover the underlying parameters[3].
- **Performance Metrics:** Evaluated using Mean Squared Error (MSE) and R-squared metrics to quantify prediction accuracy.
- **Edge Case Analysis:** Tested the model with various data conditions, including:
  - High-dimensional data.
  - Data with multicollinearity.
  - Datasets with varying noise levels.
- **Comparison with Baselines:** Compared the results against standard linear regression without regularization to demonstrate the benefits of ElasticNet.

## **What parameters have you exposed to users of your implementation in order to tune performance?**

The ElasticNet implementation exposes the following tunable parameters:

- **`alpha`**: Controls the overall strength of the regularization. Higher values impose more regularization.
- **`l1_ratio`**: Balances the contribution between L1 and L2 regularization. A value of 0 corresponds to only L2 regularization, while a value of 1 corresponds to only L1.
- **`fit_intercept`**: Boolean indicating whether to calculate the intercept for the model.
- **`max_iter`**: The maximum number of iterations for the optimization algorithm.
- **`tolerance`**: The tolerance for the optimization algorithm's convergence.
- **`learning_rate`**: Step size for gradient descent updates.
- **`random_state`**: Seed used by the random number generator for reproducibility.

These parameters allow users to fine-tune the model to achieve optimal performance based on their specific dataset characteristics.

## **Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental to the model?**

**Challenging Inputs:**

- **Highly Imbalanced Features:** Datasets where certain features dominate others in scale can affect the regularization effectiveness. Proper feature scaling is essential.
- **Non-linear Relationships:** The current implementation assumes linear relationships between predictors and the target variable. It may underperform on datasets with complex non-linear patterns.
- **Sparse Data with High Dimensionality:** While ElasticNet is suitable for high-dimensional data, extremely sparse datasets might require additional preprocessing or dimensionality reduction techniques.

**Potential Workarounds:**

- **Feature Scaling:** Implementing automatic feature scaling can mitigate issues with imbalanced feature scales.
- **Polynomial Features:** Extending the model to include polynomial or interaction terms can help capture non-linear relationships.
- **Dimensionality Reduction:** Techniques like PCA can be integrated to handle extremely high-dimensional sparse data more effectively.

With additional time, these enhancements can be incorporated to improve the model's robustness and applicability to a wider range of datasets.

## **Usage Examples**

Below are examples demonstrating how to use the implemented ElasticNet model:

### **Training the Model**

```python
from ElasticNet import ElasticNetModel
import numpy as np

# Generate synthetic data
from generate_regression_data import linear_data_generator

# Parameters for synthetic data
m = np.array([1.5, -2.0, 3.0])
b = 4.0
rnge = [0, 10]
N = 100
scale = 1.0
seed = 42

# Generate data
X, y = linear_data_generator(m, b, rnge, N, scale, seed)

# Initialize the model with desired parameters
model = ElasticNetModel(alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tolerance=1e-4, learning_rate=0.01, random_state=42)

# Fit the model to the training data
model.fit(X, y)
