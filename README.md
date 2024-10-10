## CS584: Machine Learning
### Project-1 - Linear regression with ElasticNet regularization (combination of L2 and L1 regularization)

#### Team Members:
##### A20584318 -ANSH KAUSHIK
##### A20593046 - ARUNESHWARAN SIVAKUMAR
##### A20588339 - HARISH NAMASIVAYAM MUTHUSWAMY
##### A20579993 - SHARANYA MISHRA

#### Introduction:
<b>ElasticNet</b> linear regression model combines L1 (lasso) and L2 (ridge) regularisation to handle correlated information and enhance prediction. We have added the Linear ElasticNet model in our models folder and have imported it in our test files to test the model with two dataset small_test.csv and Ames.csv(our testing dataset).

#### Questions:
<b>1. What does the model you have implemented do and when should it be used?</b>

The ElasticNet model, which combines L1 (lasso) and L2 (ridge) penalties, is a regularised linear regression model that we have implemented. By managing multicollinearity and high-dimensional data, it enhances prediction. Compared to conventional linear models, ElasticNet balances feature selection (L1) and coefficient shrinkage (L2) to produce predictions that are more stable and avoid overfitting. 

The ElasticNet function:

$$
\text{minimize} \left( \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \lambda \left( \alpha \sum_{j=1}^{p} \left| \beta_j \right| + \frac{(1 - \alpha)}{2} \sum_{j=1}^{p} \beta_j^2 \right) \right)
$$

Where:
- \( y_i \) is the actual value,
- \( \hat{y}_i \) is the predicted value,
- \( \beta_j \) are the coefficients,
- \( \lambda \) is the regularization strength,
- \( \alpha \) controls the balance between L1 (lasso) and L2 (ridge) regularization.

ElasticNet is used for high-dimensional data, where the number of features exceeds observations. It handles multicollinearity by balancing lasso (L1) and ridge (L2) penalties, improving model stability. It's effective in feature selection, regression with many correlated variables, and preventing overfitting in complex datasets.

<b>2. How did you test your model to determine if it is working reasonably correctly?</b>

We tested our model on the large Ames Dataset and achieved a R2 score of over 0.78, showing that it works well with big datasets. It's important to apply normalization, transformations (like log transformation), and one-hot encoding for categorical variables before using the model for ensuring optimal performance of the model. Furthermore, we used the small_test.csv dataset provided on the GitHub repository to test our model, and were able to obtain an R2 score of greater than 0.80. This result demonstrates that our model can effectively adapt to new datasets while maintaining high accuracy.

However, the model may not work as well with sparse or weakly related data because it relies on strong feature relationships to make accurate predictions. In cases where the data has little connection between features, the model may struggle to find clear patterns.

<b>3. What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)</b>

The parameters exposed to users in order to tune performance include:
1. The <b>alpha</b> parameter helps to prevent overfitting by adding a penalty to large coefficients.
2. The <b>l1_ratio</b> allows users to balance the benefits of L1 and L2 regularization.
3. The <b>iterations</b> and <b>learning_rate</b> provide control over the convergence of the opti

We added another dataset called Ames, where users can adjust different parameters and find the best settings. While working with this dataset, we tried different values for each parameter and tuned them to get the best R² score and the lowest Mean Squared Error (MSE). The values that worked best for us were: alpha = 0.1, l1_ratio = 0.5, iterations = 10,000, and learning_rate = 0.0001.

<b>4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?</b>

The primary challenge that the model may encounter is working with sparse and poorly correlated datasets, as outlined in Question-2. We believe this issue cannot be entirely resolved, as collinearity and a higher number of data points significantly contribute to reducing the Mean Squared Error (MSE) and increasing the R² score in linear models like Elastic Net. When a dataset is sparse and lacks meaningful correlations, it can lead to overfitting, where the model captures noise instead of the underlying patterns.

The issue is fundamental,while we cannot completely remove these issues, using techniques like regularization can help manage them to some extent, but a well-structured dataset is essential for robust model performance.

#### Our Observations and Conclusion:
#### 1. Dataset - small_test.csv:
Our training and testing datasets achieved R² scores exceeding 0.80, demonstrating strong predictive performance. Moreover, the mean squared error has decreased in the testing dataset compared to the training dataset, indicating enhanced model performance on unseen data. Additionally, the negligible mean bias reveals that the model's predictions are, on average, closely aligned with the actual target values, reflecting accurate and reliable performance. This suggests that the model does not consistently overestimate or underestimate outcomes, resulting in well-calibrated predictions

#### 2. Dataset - Ames.csv:
Our model gives R2 score of around 0.78 for train and 0.80 for test. Also, as we can see the bias is almost negligible hence our code works fine. The bias analysis indicates that our model exhibits negligible bias, with the mean bias close to zero. The residuals are randomly scattered around the horizontal line, suggesting no systematic underprediction or overprediction. This confirms that the model is well-calibrated and performs reliably.