Team Members:
1. Badri Adusumalli A20530163
2. Bhuvana Chandrika Natharga A20553587
3. Santhosh Kumar Kathiresan A20546185
4. 


## 1. What does the model you have implemented do, and when should it be used?

The model implemented is a **Linear Regression model with ElasticNet regularization**. ElasticNet is a regularized linear regression technique that combines **L1 (Lasso)** and **L2 (Ridge)** regularization. The primary purpose of this model is to **predict a continuous target variable** (regression tasks) based on a set of input features.

### ElasticNet Explained:
- **Linear Regression** predicts the target variable $y$ using a linear combination of the input features $X$. In mathematical terms, this can be written as:
  
<p align="center">
  <img src="https://github.com/user-attachments/assets/aa29a395-4b9e-4967-958d-375bc62810fe" alt="Formula" />
</p>

  where $\beta_0$ is the intercept, and $\beta_i$ are the coefficients for each feature $X_i$.

- **Regularization** adds a penalty to the model to prevent overfitting. ElasticNet combines the penalties from both L1 and L2 regularization:

<p align="center">
  <img src="https://github.com/user-attachments/assets/43e0ab95-0b9f-461f-9778-e8c4be119637" alt="ElasticNet Formula" />
</p>

  Here:

  - **L1 Penalty (Lasso)**: $\lambda_1 \|\beta\|_1$ encourages sparsity by driving some coefficients to zero.
  - **L2 Penalty (Ridge)**: $\lambda_2 \|\beta\|_2^2$ shrinks the coefficients to reduce model complexity.

### When to Use ElasticNet:
- **Multicollinearity**: When features are correlated, ElasticNet performs better than standard linear regression or Lasso because it can handle correlated predictors well.
- **Feature Selection**: ElasticNet can be used to perform feature selection by shrinking some coefficients to zero (like Lasso).
- **Overfitting Prevention**: If you have many features or high-dimensional data, ElasticNet helps prevent overfitting by adding regularization to the model.


### 2. **How did you test your model to determine if it is working reasonably correctly?**

The testing procedure involves the following steps:

1. **Loading and Preprocessing the Data**:
   - The code reads a dataset from a CSV file (`small_test.csv`). The dataset is assumed to have a target variable column named `y` and some feature columns, which may be numerical or categorical.
   - Categorical features are converted to numerical values using **OneHotEncoder**. If the target variable is categorical, it is encoded using **LabelEncoder**.

2. **Splitting the Data into Training and Testing Sets**:
   - The data is split into **80% training** and **20% testing**. This allows the model to learn from the training data and then evaluate its performance on the unseen test data.

3. **Training the Model**:
   - The **ElasticNetModel** class is used to fit the training data. The model learns the relationship between the features and the target variable using a coordinate descent algorithm.

4. **Making Predictions on the Test Data**:
   - After training, the model makes predictions on the test set.
   - If the original target variable was categorical, the predictions are converted back to categorical labels.

5. **Evaluating Model Performance**:
   - Two metrics are used for evaluation:
     - **Mean Squared Error (MSE)**: Measures the average squared difference between the actual and predicted values. A lower MSE indicates better performance.
     - **R-squared (R²)**: Represents the proportion of variance in the target variable that is explained by the features. An R² closer to 1 indicates a better fit.

6. **Printing Model Parameters**:
   - The code also prints the learned coefficients, intercept, and the number of non-zero coefficients. This helps understand the impact of each feature and the regularization effect.

### 3. **What parameters have you exposed to users of your implementation in order to tune performance?**

The model exposes several parameters to tune the performance:

1. **`alpha` (float)**:
   - This parameter controls the overall strength of the regularization. A higher value of `alpha` increases the regularization effect, leading to smaller coefficients and potentially better generalization to unseen data.

2. **`l1_ratio` (float)**:
   - This parameter controls the mix between **L1 (Lasso)** and **L2 (Ridge)** regularization. It should be between 0 and 1.
     - `l1_ratio = 1` corresponds to pure Lasso regularization.
     - `l1_ratio = 0` corresponds to pure Ridge regularization.
     - Values between 0 and 1 provide a mix of L1 and L2 regularization.

3. **`tol` (float)**:
   - This is the tolerance for the stopping criteria. It controls how small the change in coefficients should be before the optimization process stops. A smaller tolerance can lead to a more precise solution but may take longer to converge.

4. **`max_iter` (int)**:
   - This parameter specifies the maximum number of iterations for the optimization process. Increasing this value can help the model converge if the optimization is not reaching the desired tolerance.

These parameters provide users with the ability to customize the model's regularization behavior and optimization settings.

### 4. **Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental to the model?**

Yes, there are specific scenarios where this implementation may face challenges:

1. **Large Sparse Matrices**:
   - The current implementation uses **NumPy arrays** and is not optimized for handling large sparse matrices. If the input data is sparse, it would require significant memory and computation. Using a library like **SciPy** for sparse matrix support could help optimize the code for such cases.

2. **Multi-target Regression**:
   - The current implementation assumes a single target variable (regression with one output). If the problem requires predicting multiple targets simultaneously, the code would need to be extended to handle multi-output regression.

3. **Categorical Data Handling**:
   - The implementation uses **OneHotEncoder** for categorical features, which can lead to a large number of features if there are many categories. For high-cardinality categorical variables, alternative techniques such as **target encoding** or **embedding representations** could be used to reduce the dimensionality.

4. **Non-linearity**:
   - ElasticNet is a linear model, meaning it assumes a linear relationship between features and the target. For datasets where the relationship is highly non-linear, this model may not perform well. Using kernel methods or non-linear models like **decision trees** or **neural networks** would be more suitable for such cases.

### Possible Improvements:
- **Support for Sparse Data**: Implement support for sparse matrices using libraries like **SciPy**.
- **Kernel Extension**: Add support for kernel methods to handle non-linear relationships.
- **Handling Multi-output Regression**: Extend the code to support predicting multiple target variables.
- **Advanced Encoding Techniques**: Use more sophisticated encoding techniques for high-cardinality categorical variables.

### Summary:
The implemented **ElasticNet** model is a versatile linear regression technique that combines both **Lasso** and **Ridge** regularization. It can be tuned to handle overfitting and perform feature selection. The code supports handling categorical features, evaluating performance with appropriate metrics, and tuning important parameters for optimal results. However, it has some limitations, such as handling sparse data and multi-target regression, which could be addressed with further improvements.
