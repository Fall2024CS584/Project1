# Project 1 - ElasticNet Model

Name: Gurjot Singh Kalsi     CWID: A20550984

Name: Siva Vamsi Kolli   CWID: A20560901

Name: Sai Teja Reddy Janga     CWID: A20554588

This project implements an ElasticNet regression model, a linear regression method that combines both L1 (Lasso) and L2 (Ridge) penalties. The model is particularly useful when dealing with datasets that have multicollinearity (high correlation among features) and when you need to perform feature selection by shrinking some coefficients to zero.

## 1. What does the model do and when should it be used?

### What:
The ElasticNet regression model estimates the relationships between a dependent variable (target) and multiple independent variables (features) while applying regularization to prevent overfitting. The model combines the strengths of Lasso and Ridge regression:
- **Lasso** (L1) for feature selection, as it can shrink coefficients of some features to zero.
- **Ridge** (L2) for minimizing the impact of multicollinearity and stabilizing the model.

### When to Use:
ElasticNet is best used in scenarios where:
- You have a large number of features, many of which might be irrelevant or redundant.
- There is high multicollinearity (i.e., correlation between input features).
- You want a model that can perform both feature selection and regularization.

It is particularly effective for datasets where neither pure Lasso nor Ridge performs optimally due to the characteristics of the data.

## 2. How did you test your model to determine if it is working reasonably correctly?

### How:
The model was tested using the following methods:
1. **Synthetic Data**: Data was generated using a separate script (`generate_regression_data.py`) to create a dataset with known coefficients and noise. The model was then trained on this synthetic data, and the results were compared to the expected coefficients and predictions.
2. **Mean Squared Error (MSE) and Mean Absolute Error (MAE)**: These metrics were computed after training the model, and thresholds were used to validate the performance of the model. For example, the test script checks that the MSE and MAE are below a certain value.
3. **Visual Validation**: The predictions of the model were plotted against actual values to visually inspect how well the model fits the data. Residuals were also plotted to check if they are evenly distributed, indicating a well-fitting model.

## 3. What parameters have you exposed to users of your implementation in order to tune performance?

### Exposed Parameters:
The following parameters can be tuned by users to control the model's performance:
- **alpha**: This controls the overall strength of regularization. A higher value applies more regularization (both L1 and L2).
- **rho**: This balances the ratio between L1 (Lasso) and L2 (Ridge) penalties. `rho=0` applies only L2 regularization, `rho=1` applies only L1, and values in between apply both.
- **max_iter**: This defines the maximum number of iterations the model will take to converge during optimization.
- **tol**: The tolerance for optimization. Smaller values can lead to more accurate solutions but may take longer to compute.

## 4. Are there specific inputs that your implementation has trouble with?

### Problematic Inputs:
The model might struggle with:

- **Nonlinear relationships**: Since the model is fundamentally linear, datasets with nonlinear patterns cannot be accurately modeled without feature engineering (e.g., polynomial features or transformations).
- **Highly imbalanced data**: The model assumes that the noise in the data is homoscedastic (having the same variance), and imbalanced data can affect its performance by distorting this assumption.

### Potential Workarounds:
- **Nonlinearity**: To handle nonlinearity, feature engineering could be applied, or a more complex model like decision trees or neural networks might be used.
- **Imbalanced Data**: Techniques like resampling (oversampling/undersampling) or adjusting model weights could help in mitigating the effects of imbalanced data.

## 5. Steps to run the code. 

The project includes a script for generating synthetic regression data, which can be used to test the ElasticNet model. The script generates data based on a user-specified linear equation with noise.

### Command to Generate Data:
```bash

python3 Project1/generate_regression_data.py -N 100 -m 3 -2 -b 5 -scale 0.1 -rnge -10 10 -seed 42 -output_file Project1/elasticnet/models/small_test.csv

### Explanation of Arguments:
- `-N 100`: Specifies the number of samples to generate.
- `-m 3 -2`: Defines the slope coefficients for the linear relationship (in this case, two features with slopes of 3 and -2).
- `-b 5`: Sets the intercept (offset) of the linear equation.
- `-scale 0.1`: Adds Gaussian noise with a standard deviation of 0.1.
- `-rnge -10 10`: Specifies the range from which feature values (X) are uniformly sampled.
- `-seed 42`: Sets the random seed for reproducibility.
- `-output_file`: Specifies the path to save the generated dataset as a CSV file.

### Example Output:
The command will generate a CSV file (`small_test.csv`) with columns for each feature (`x_0`, `x_1`) and a target value (`y`). This file can then be used to train and test the ElasticNet model.

Run the ElasticNet.py file to train the model.

Then run the test_ElasticNetModel.py to test the model.

Results shows MSE and MAE values and regression plots. 

### 6. Project Structure
- **ElasticNet.py**: Contains the implementation of the ElasticNet regression model.
- **metrics.py**: Provides functions for evaluating model performance (MSE, MAE).
- **generate_regression_data.py**: A script to generate synthetic linear data with noise for testing the model.
- **test_ElasticNetModel.py**: A test script to evaluate the model.