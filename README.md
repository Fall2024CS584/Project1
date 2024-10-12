# Project 1 

# ElasticNet Regression Model

## Dhruv Singh - A20541901
## Kunal Samant - A20541900
## Introduction

In this project, I implemented an ElasticNet regression model from scratch. ElasticNet is a type of linear regression that combines both Lasso (L1) and Ridge (L2) regularization methods. It's particularly useful when dealing with datasets that have lots of features, some of which might be highly correlated, or when we want to perform feature selection. ElasticNet is beneficial when there are many features and some are correlated, or when neither Lasso nor Ridge regression alone provides optimal results. I chose to create my own implementation without relying on big libraries like scikit-learn or pandas to gain a deeper understanding of how the algorithm works under the hood.

## Testing the Model

To ensure my model works correctly, I conducted several tests. First, I generated synthetic data using NumPy, where I knew the underlying relationship between features and the target variable. After training the model on this data, I compared the coefficients learned by my model to the actual coefficients. This helped me see if the model could recover the true relationship.

Next, I used real datasets, such as the Boston Housing dataset, which predicts house prices based on features like the number of rooms, crime rate, and more. I also tried a custom house prices dataset that included categorical data like city names. For these datasets, I manually preprocessed the data, handling missing values and encoding categorical variables since I wasn't using pandas. I standardized the features manually using NumPy, trained the model, and made predictions. I calculated the Mean Squared Error (MSE) manually to assess how well the model performed and plotted the actual versus predicted values to visualize the results.

Additionally, I compared my implementation with scikit-learn's ElasticNet model by running the same datasets through both models. By comparing the outputs, I could check if my implementation was on the right track. I also tested the model with inputs that might cause problems, like features with zero variance or highly correlated features, to ensure the model is robust.

## Parameters for Tuning

I exposed several parameters so users can adjust the model according to their needs. The alpha (`alpha`) parameter controls the strength of regularization; higher values mean more regularization. The L1 ratio (`l1_ratio`) determines the mix between Lasso and Ridge regression, with a value of 1.0 meaning Lasso only and 0.0 meaning Ridge only. The maximum iterations (`max_iter`) parameter sets the maximum number of iterations the algorithm will run; more iterations can lead to better convergence but will take longer. The tolerance (`tol`) determines when to stop the algorithm based on the change in coefficients; smaller values can lead to more precise models but might take more time. I experimented with these parameters to see how they affect the model's performance. For example, changing `alpha` and `l1_ratio` can help prevent overfitting or underfitting.

## Challenges Faced

During the project, I faced several challenges. When I tried to convert the data to floats, I ran into errors because some features were strings, like city names. Since I wasn't using pandas, I had to manually implement one-hot encoding using NumPy. This was a bit tricky and made the code longer. With more time, I might look into ways to handle categorical variables more efficiently, or perhaps use small libraries dedicated to encoding.

Another challenge was data preprocessing. Without libraries like pandas, reading and cleaning data was more challenging. I used Python's built-in CSV reader and NumPy for data manipulation. While it was a good learning experience, using pandas would have made this process faster.

Implementing the ElasticNet algorithm itself was also challenging. Ensuring that the coordinate descent algorithm, which ElasticNet uses, was correctly implemented required me to read up on the algorithm and carefully code it, testing along the way. It was satisfying to see it work, but it required careful attention to detail.

Handling zero variance features posed another problem. When a feature had the same value for all samples, the standard deviation became zero, leading to division by zero errors during standardization. I set any zero standard deviations to one during standardization, effectively leaving those features unchanged. This works, but in practice, such features might be removed since they don't provide useful information.

## Limitations and Future Work

There are some limitations to my implementation. If a categorical variable has many unique values, one-hot encoding increases the number of features significantly, which can be problematic. In the future, I might explore other encoding methods, like target encoding, to handle this issue.

ElasticNet is a linear model, so it might not capture non-linear relationships well. Given more time, I could look into adding polynomial features or using other models to address this limitation. The coordinate descent algorithm can be slow for large datasets, so optimizing the code or using more efficient algorithms could help improve performance.

## Execution Command
pytest elasticnet/tests/test_ElasticNetModel.py

## Conclusion

Implementing ElasticNet from scratch was a challenging but rewarding experience. I learned a lot about how the algorithm works and the importance of data preprocessing. While there are still areas for improvement, especially in handling categorical data and optimizing the algorithm, I'm satisfied with the progress made. In the future, I plan to automate more of the data preprocessing steps, explore ways to handle large numbers of categorical variables, and optimize the algorithm for better performance.

