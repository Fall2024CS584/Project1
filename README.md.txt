# Elastic Net Regression on Boston Housing Dataset

## Project Overview

This project implements Elastic Net regression, a regularization technique that combines L1 (Lasso) and L2 (Ridge) regularization methods, to predict housing prices in Boston. Elastic Net is particularly useful when dealing with datasets that may have multicollinearity, as it encourages sparsity in the model coefficients and can handle situations where the number of predictors is greater than the number of observations.

### Objectives

- To predict the median value of owner-occupied homes in Boston.
- To demonstrate the application of Elastic Net regularization in linear regression.
- To optimize hyperparameters using cross-validation to improve model performance.

## Dataset

The dataset used in this project is the **Boston Housing dataset**, which contains various features about houses in Boston and their prices. The dataset can be downloaded from [Kaggle - Boston Housing](https://www.kaggle.com/c/boston-housing).

### Features

The dataset consists of the following attributes:

- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq. ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centers
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: \(1000(Bk - 0.63)^2\) where \(Bk\) is the proportion of Black residents by town
- **LSTAT**: Percentage of lower status of the population
- **MEDV**: Median value of owner-occupied homes in $1000s (target variable)

## Installation

To run this project, you'll need to have Python and Jupyter Notebook installed on your machine. Follow these steps to set up the environment:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ZaibN/mlP1.git

2. Navigate to the project directory:
	cd ElasticNet_Project
3. Create a virtual environment (optional):
	python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


Usage
Open the Jupyter notebook: You can start Jupyter Notebook from the terminal:

This will open a new tab in your web browser.

Run the notebook cells sequentially: Open the EAR.ipynb notebook and execute each cell in order. The notebook is organized to guide you through the entire modeling process, from data preprocessing to model evaluation.


Hyperparameter Tuning
To enhance the model's performance, hyperparameter tuning was conducted to find the optimal values for alpha (the regularization strength) and l1_ratio (the mix ratio between L1 and L2 regularization). The tuning was performed using cross-validation on the training dataset.

Best Parameters Found:
Alpha: 10.0
L1 Ratio: 1.0 (indicating full L1 regularization)


Results
The final Elastic Net model achieved a Mean Squared Error (MSE) of 76.79 on the test dataset, indicating improved predictive performance compared to a baseline model without regularization. This demonstrates the effectiveness of Elastic Net in handling multicollinearity and preventing overfitting.

Model Performance Metrics
Train MSE: 20.50 
Test MSE: 76.79
R² Score: 0.80 

Visualization
The notebook includes visualizations such as:
Scatter plots comparing actual vs predicted values.
Residual plots to check the distribution of errors.


Future Work
Experiment with different datasets to evaluate the model's robustness.
Implement additional regularization techniques, such as Ridge regression, for comparison.
Conduct feature engineering to create new predictors from existing features for potentially better model performance.

