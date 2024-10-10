# Project 1: Linear Regression with ElasticNet Regularization

**Course:** CS584 - Machine Learning <br>
**Instructor:** Steve Avsec<br>
**Group Menbers:**
- ssaurav@hawk.iit.edu (FNU Saurav)
- psavant@hawk.iit.edu (Pallavi Savant)

## Project Overview
Linear regression with ElasticNet regularization (combination of L2 and L1 regulariza-
tion)

## Key Features
- Linear Regression model built from scratch
- Incorporation of ElasticNet Regularization (L1 + L2 penalties)
- Efficient gradient descent implementation for optimization
- Model evaluation using MSE (Mean Squared Error)
- Comparision of performance with other regression techniques

## Requirements
- **Python 3.x**
- **NumPy**: For numerical operations
- **Pandas**: For dataset manipulation
- **Matplotlib**: For plotting results (optional) <br>

**Install the necessary libraries using below command:**

```bash
pip install numpy pandas matplotlib
```

## Usage
 ```bash
 Python main.py
 ```

## Model Functions:
- **fir(X, Y):** Trains the model on training data.
- **predict(X):** Predcits target values for the test data.
- **find_optimal_penalties(X_train, Y_train, X_test, Y_test):** Performs gird search to find the optimal L1 and L2 penalties for Elastic Model.

## Explanation of the Model

**1. What does the model you have implemented do and when should it be used?**
- The model we have implemented is a linear regression model with Elastic regularization, which integrates both L1(Lasso) and L2(Ridge) techniques.
- This approach helps reduce overfitting by regulating the size and sparsity of the coefficients.
- It is especially beneficial when we need balance between L1(which helps in feature selection) and L2(which provides smoother regularization), enhancing the performance on the datasets with irrelevant features.
- Elastic net is especially effective in high-dimentional datasets where traditional linear regression 
may struggle with overfitting or irrelevant variables.


**2. How did you test your model to determine if it is working reasonably correctly?**
- We evaluated our model by training it on a dataset that predicts salary based on years of experiences.
- To verify the models ability to generalize, we have divided the data into training sets and testing sets.
- We used MSE (Mean Squared Error) to test set to measure the performance and then we compared it with the linear regression model to determine how ElasticNet regularization enhances prediction accuracy.
- Additionally, we also performed grid search to find the optimal L1 and L2 penalty values, fine-tuning the model and   ensuring it behaves as expected across different regularization strengths.


**3. What parameters have you exposed to users of your implementation in order to tune performance?**
The parameters available in my implementation includes:
- learning_rate: Which Regulates the step size during gradient descent.
- iterations: Specifies the number of gradient descent optimization cycles.
- l1_penalty: sets the intensity of L1 regularization (Lasso)
- l2_penalty: Sets the intensity of L2 regularization (Ridge)


**4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental to the model?**
- Our implementation may encounter difficulties with datasets exhibiting significant high noise levels.
- Datasets with excessive noise might result in unstable coefficient estimates.
- If given more time, we would consider using more advanced optimization methods to better manage such cases.

## Output Example:
```yaml
Optimal L1 penalty: 0.0001
Optimal L2 penalty: 0.0001
Best Mean Squared Error for Elastic Net: 21026060.7492401
Linear Regression Predictions: [ 40594.69 123305.18  65031.88]
Elastic Net Regression Predictions: [ 40835.13 123079.36  65134.56]
Real values: [ 37731. 122391.  57081.]
Linear Regression Trained W: 9398.92
Elastic Net Regression Trained W: 9345.94
Linear Regression Trained b: 26496.31
Elastic Net Regression Trained b: 26816.23
```

## Future Improvements:
- Test the model with multiple complex datasets.
- Implement feature scaling and normalization as part of preprocessing steps.