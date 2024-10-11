# Project 1
-----------------------------------------------
Group members:
A20594429 - Satya mani srujan Dommeti
A20543669 - SAI VENKATA VAMSI KRISHNA YELIKE
A20583607 - Akshitha Reddy Kuchipatla
A20577962 - Arjun Singh
-----------------------------------------------
* **What does the model you have implemented do and when should it be used?**  
  I have implemented an ElasticNet linear regression model. This model is used when you need a combination of L1 (Lasso) and L2 (Ridge) regularization techniques to handle multicollinearity, high-dimensional datasets, or when the data may have irrelevant features that should be eliminated (Lasso effect). It is best used when both sparsity (zero coefficients) and smoothness (small coefficients) are desired.

* **How did you test your model to determine if it is working reasonably correctly?**  
  I tested the model by fitting it on a dataset with both features and target values and verifying the predicted values. I also implemented convergence plots, R² scores, and MSE calculations to assess the accuracy of the predictions. Additionally, I validated the model's performance using different evaluation plots like predicted vs. actual values and feature importance.

* **What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)**  
  The model exposes several tunable parameters:  
  - `alpha`: Regularization strength (higher values increase regularization)  
  - `l1_ratio`: Determines the balance between L1 and L2 regularization (0 is Ridge, 1 is Lasso)  
  - `max_iter`: Maximum number of iterations for gradient descent  
  - `tol`: Tolerance for convergence  
  - `learning_rate`: Step size for gradient updates  
  Example usage:
  ```python
  model = ElasticNetModel(alpha=0.01, l1_ratio=0.7, max_iter=2000, tol=1e-4, learning_rate=0.05)
  results = model.fit(X, y)
  predictions = results.predict(X)
  ```

* **Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?**  
  The model may have trouble with highly sparse datasets or very small feature sets where L1 regularization can zero out too many coefficients. It also might struggle with datasets where features are on very different scales, which could affect the gradient descent process. With more time, I could address this by adding automatic feature scaling (e.g., standardization) and adjusting the learning rate dynamically. These issues are not fundamental and could be resolved with further enhancements.

