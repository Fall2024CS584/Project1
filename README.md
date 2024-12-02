Rudraksha Ravindra Kokane 
rkokane@hawk.iit.edu
A20586373  

The project implements a ElasticNet model. ElasticNet is regularised linear REgression model, with L1 and L2 regularization techniques combined. 

Elastic Net Objective function L(w) = (1/N) Σ (i=1 to N) (yi - xi^T w - b)^2 + α [ρ ||w||₁ + (1 - ρ) ||w||₂²]

where, L(w): Elastic Net loss function.
N: Number of data points.
w: Weight vector (model coefficients).
b: Bias term (intercept).
xi: Feature vector of the ith data point.
yi : Actual target value for ith data point
​
∥w∥1: L1 norm of the weights (Lasso penalty).
∥w∥2 : Squared L2 norm of the weights (Ridge penalty).
α: Regularization strength (scales the combined penalty terms).
ρ: Mixing parameter (0≤ρ≤1):
ρ=1: Pure Lasso regularization.
ρ=0: Pure Ridge regularization.
0<ρ<1: A combination of Lasso and Ridge.


What does the model you have implemented do and when should it be used?
Elasticnet model is linear regression model with L1 and l2 regularization techniques combined. This model perform feature selection by adjusting weights and bias to prioritize relevant features. REgularization in model, assists in preventing overfitting.
This model is helpful in case of datasets with many features, expecially in case they are corelated. The model is useful only for cases of Linear Regression. Model is used to avoid overfitting in case of linear regression. 

How did you test your model to determine if it is working reasonably correctly?
Constructing model was based loosely on Scikit Learns ElasticNet Implementation. The testing was carried by training and testing model on california housing dataset from scikit-learnsw dataset repo. The resultant MSE for testing set was 0.6369, which is indication of how well model performed. The performance was also visualised to verify reliability of model. The model was also tested against sample data with self created record to test for testing the model.

What parameters have you exposed to users of your implementation in order to tune performance?
There are four parameter for tuning performance:
alpha: which is used to determine regularization strength
l1_ratio: It balances the L1 and L2 regularization factors
learning_rate: determine step size in gradient Descent
max_iters: Gives maximum iterations required for optimizing the model performance.

Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental to the model?
The model works only in cases whre relationship between data and target variable is linear. In case of extreme non linearity in data, the model is less efficient compared to any complex model. Also model does not have method to assist in hyperparameter Optimization techniques like Grid SearchCV. Using complex tehcniques than Gradient Descent like in sklearn Linear models will allow model to perform better.
SWe can deal with some of the issues given sufficinet time and resources.

The directory has three files:
Readme.txt file 
Linear Regression ElasticNet Regularization.ipynb
ElasticNetReg.py

ElasticNetReg.py hosts the model, which has three methods.
    Fit method is used to train the model, and calculate weights and biases.
    Predict function predicts the target values, according to results of fit method.
    Calculate loss function calculates the mse for each iteration and combines it with regularization penalties. it helps in tracking training progress. It provides insights into how model avoid overfitting, while fitting data and minimizing errors.

The model is used on California Housing dataset, to train and test model, and verify its use. The model performs good, and with improvements mentioned above, it can results in better prediction accuracy avoiding errors and overfitting.

The notebook also visualize the data of how predicted and actual values compares.
