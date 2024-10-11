# Project 1 #Team Members
    #Bezawada Sai Sravanya - A20561552
    #Karuturi Veerendra Gopichand - A20529571
    #Praveen Kumar Gude - A20546614
    #Abharnah Rajaram Mohan - A20546967


## What does the model you have implemented do and when should it be used?

The implemented model is Elastic Net Regression. Elastic Net combines both L1, known as Lasso, and L2, known as Ridge regularization, making it useful for cases with many correlated features or when the number of features exceeds the number of observations. This helps to prevent overfitting while maintaining interpretability of the coefficients. It also evaluates their performance using the metrics Mean squared error(MSE) and R-squared(R2)
This model is used to understand the regularization when exploring L1 and L2 penalities affect model performance.

## How did you test your model to determine if it is working reasonably correctly?

We have run my model on the synthetic datasets in which relationships are known. Scatter plots comparing predicted versus true values were created for each model.The performance of each model was compared using their respective R2 scores and MSE. This helps to determine which model performs best under given conditions.

## What parameters have you exposed to users of your implementation in order to tune performance?

This provides the user with sensitivity to key parameters in the implementation and allows users to tune for optimal model performance. For this,tuning of alpha controls the general strength of the regularization applied to the model and tuning of l1_ratio controls the mix of L1 and L2 penalties applied.

## Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental to the model?

The implementations may suffer when the datasets are highly imbalanced, including multi collinearity which can destabilize coefficient estimates.
Implement techniques such as Principal Component Analysis (PCA) to reduce dimensionality or use feature selection methods to eliminate redundant features. However, this doesn't fundamentally change the nature of the models.

