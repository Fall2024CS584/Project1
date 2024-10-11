# Project 1 


# Credit Spread Linear-regression-with-ElasticNet-regularization - CS584 Machine Learning

[Project Git rep](https://github.com/PaulAnrd/Credit-Spread-Linear-regression-with-ElasticNet-regularization/tree/main)

[Subject](https://github.com/Fall2024CS584/Project1)

Team :
- David Bettane
- Gaspard Devoivre
- Yahya Mohammed Y Alqahtani 
- Paul Ainardi


### 1. What does the model you have implemented do and when should it be used?
Elastic Net is a linear regression model that combines the properties of both Lasso and Ridge regularization techniques. The first technique tends to produce sparse models with fewer coefficients, effectively performing feature selection, while the other one tends to distribute the coefficients more evenly, reducing the risk of overfitting. The code uses gradient descent to iteratively update the model parameters, and the stopping condition ensures that the algorithm halts when the change in weights becomes sufficiently small.

Elastic Net is particularly useful when you have more features than samples or when many features are correlated. In such cases, Lasso can arbitrarily pick one among correlated features, while Elastic Net tends to distribute the weights among them. Moreover, the combination of L1 and L2 regularization helps in controlling overfitting by shrinking the coefficients in a balanced manner. Finally, if we suspect that only a subset of the features are relevant, we would definitely use Elastic Net, because it can help by setting some coefficients to zero and retaining others with reduced impact.




### 2. How did you test your model to determine if it is working reasonably correctly?
First of all, we made a comparison between y and y_pred, which involves comparing the actual target values (y) with the predicted values (y_pred) produced by the model. A close alignment between these values generally indicates a well-performing model. Then we plotted the predicted values with the actual values. Finally, we made a histogram of residuals to check the distribution of errors


### 3. What parameters have you exposed to users of your implementation in order to tune performance?
These parameters are exposed to the user in this implementation, so that we can tune the performance of the ElasticNet model. The alpha parameter determines the overall strength of regularisation, i.e., increasing this parameter makes it more regularised and so lower values for the model coefficients are produced, reducing over-fitting. The l1_ratio parameter determines the balance between L1 (Lasso) and L2 (Ridge) regularisation. The closer this value gets to 1, the more L1 regularisation is used, which aims for feature selection. The closer it is to 0, the more Ridge regularisation is used. This means that large coefficient values can be stopped from happening but they won’t be forced to zero. The max_iter parameter determines the maximum number of iterations that the gradient descent is performed on and makes a difference to whether the model converges within those iterations or not. The tolerance value tol determines when to stop optimising the model – if the change in the model weights between iterations falls below the tolerance, the algorithm stops. So a smaller value for tol gives a more precise model, at the cost of more computation being needed.


### 4. Are there specific inputs that your implementation has trouble with? Given more time,could you work around these or is it fundamental to the model?
