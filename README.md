
#  Linear Regression with ElasticNet Regularization 
## Combination of L2 and L1 Regularization
### Project - 1








#### Team Members:

-     Vamsi Krishna Chitturi (A20539844) (vchitturi1@hawk.iit.edu)
-     Likhith Madhav Dasari (A20539604) (ldasari2@hawk.iit.edu)
-     Leela Lochan Madisetti (A20543643) (lmadisetti@hawk.iit.edu)
-     Santhosh Ekambaram (A20555224) (sekambaram@hawk.iit.edu)



For predicting continuous outcomes based on input features, Linear Regression is one of the main fundamental concept. But when it comes to large number of features  performance may differ and it may struggle with highly correlated features also. Which may lead to overfitting of data. To overcome this issue regularization methods are used by adding penalty terms to the model objective functions.ElasticNet regularization is one of the technique which can be used  and it is the combination of both L1 and L2 regularization. When it comes to L2(Ridge regularization), it shrinks large coefficients and not the other hand L1(Lasso regularization) will force the coefficients to exactly zero. Which effectively performs feature selection. When it comes to visualization Notebook.ipynd will gives to line plot and scattter plot for better undestanding of the model performance.

### Data Genaration CMD and DataSets:

#### test.csv

```bash
python generate_regression_data.py -N 50 -m -1 2 -3 -b 3.5 -scale 1.2 -rnge -10 10 -seed 8657309 -output_file elasticnet/tests/test.csv
```
#### data.csv

```bash
python generate_regression_data.py -N 50 -m $(printf "%s " $(seq -1 -1 -50)) -b 3.5 -scale 1.2 -rnge -10 10 -seed 8657309 -output_file elasticnet/data.csv
```
#### pima-indians-diabetes.data.csv

```bash
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```

#### And another dataset WHITE WINE is used loacally using the csv file named `winequality-white.csv`


### What does the model you have implemented do and when should it be used?

The implemented model ElasticNet which is in the form of Linear Regression, which is the combination of both L1 and L2 regularization. This is the extension of fundamental Linear regression model which controls the magnitude of the coefficients. Which provides below:

      1. Feature selection (Lasso regularization), which is setting some coefficients to zero 
      2. Reduces the impact of multicollinearity ( Ridge regularization), represents correlation between features

So, by using these two penalties, the ElasticNet model will predict continuous outcome by learning  the relationship between input features (X) and target variables (y). Which is like a standard Linear Regression. This model can be used  when the data is:

    1. High Dimensional : When more number of features are available then needed
    2. MultiCollinearity : Highly correlated features 
    3. Feature Selection : When it is needed to select features that are need and use relationship among the remaining features 
    4. Overfitting : To improve the generalization of the model, for smaller datasets

### How did you test your model to determine if it is working reasonably correctly?

 To know the working performance, following steps are taken for testing and validation:

1. `Performance Metrics`: MSE (Mean Squared Error) and R^2 score are used to measure the performance of each model (L1,L2,Elastic Net,KNN). Where MSE shows the average squared difference of predicted value and actual target values. Which talks about predictive performance. On the other hand, R^2 shows the proportion of variance in the target variable. 

2. `Visualization`:   Scatter Plot is used to show the visual validation where to visualize how closely the model predictions align with the actual target value. 

3. ElasticNet Model is `Compared with base model` like L1, L2 and Linear regression, which helps to know the improvement.

4. `K-Fold Cross Validation` has been implemented on ElasticNet model, which was used to evaluate the performance of the model by splitting the dataset by K folds, K-1 folds are used to training and testing on remaining folds. Where the process is repeated  K times. And for each fold MSE and R^2 is calculated for better insights about the data.

### What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)


There are several parameters that are exposed for tuning and model performance. 

Exposed Parameters:

1. `alpha` :  

Which controls the  overall strength of regularization. When the value is high, which increases the amount of regularization applied for the model. Which helps to prevent Overfitting the data 

Usage :
```bash
 ElasticNetModel(lambda_l1=0.01, lambda_l2=0.01, alpha=0.001, num_iterations=1000)
 ```

2. `lambda_l1 and lambda_l2`:  These are similar to l1_ratio in ElasticNet implementation. Which gives the balance between the L1 and L2 regularization models.

Usage :
```bash
# More L1 regularization (sparsity)
elastic_net = ElasticNetModel(lambda_l1=0.1, lambda_l2=0.01, alpha=0.001, num_iterations=1000)

# Balanced regularization
elastic_net = ElasticNetModel(lambda_l1=0.01, lambda_l2=0.01, alpha=0.001, num_iterations=1000)

# More L2 regularization (smoothness)
elastic_net = ElasticNetModel(lambda_l1=0.01, lambda_l2=0.1, alpha=0.001, num_iterations=1000)
```

3. `num_iterations` :  Used for maximum number of iterations for model training.

Usage: 
```bash
 elastic_net = ElasticNetModel(lambda_l1=0.1, lambda_l2=0.01, alpha=0.001, num_iterations=1000)
 ```

4.    `l1_ratio`:   This defines the balance between L1 and L2 . This is used in Elastic net model used in K-fold cross validation where L1 and L2 model are for used for calculation.

* If l1_ratio = 0, it acts as Ridge regression (only L2 regularization).
* If l1_ratio = 1, it acts as Lasso regression (only L1 regularization).
* Values between 0 and 1 give a mix of both L1 and L2 regularization.

Usage:
```bash
def __init__(self, alpha=1.0, l1_ratio=0.5, learning_rate=0.01, max_iter=1000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.coef_ = None
```

5. `CV for ElasticNet Model and KNN Model`:
 In the implementation of KNN we are taking k value as 3 as default and can be assigned as needed in function call 
 In the K-fold cross validation n_splits is used for to show the no of splits has to done for the data 

Usage :
```bash
k_fold_cross_validation(elastic_net, X, y, n_splits=5)

KNNModel(k=5)
```
### Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

Based on the nature of input data, specific challenges an be faced by the ElasticNet model. Given time so many input issues can be solved based on the dataset.

1. `Handling Missing values`:

Usage :
```bash
# Handling missing values 
data = pd.read_csv('winequality-white.csv', sep=';')
data = data.dropna()  # Remove rows with missing values
```
This include preprocessing steps for handling missing values:

- Removing rows with missing values.
- Imputing missing values with the mean or median of the respective feature.
- Using techniques such as k-nearest neighbors for imputation.


2.   `Feature Scaling`:

When it comes wine quality dataset which contains some features that has to be scaled. For instance alcohol content vs acidity. These are not scaled, which leads to poor performance. And the diabetes dataset, features like body mass index and blood pressure, which can cause issue during optimization. Without using sklearn.preprocessing library  it takes some time for implementation. Which needs in-depth understanding of data.

3.   `Multicollinearity`:

In the wine quality dataset, features like alcohol content and density may be highly correlated. This multicollinearity can make the model's coefficients unstable. In the diabetes dataset, features like skin thickness and insulin levels could also exhibit multicollinearity. By doing this it conduct a correlation analysis and removes highly correlated features.Use regularization methods (like ElasticNet) which can help mitigate the effects of multicollinearity.

4. ` Outliers`:

The wine quality dataset may have outlier like sulfur dioxide levels. Which may have skew results. And in diabetes dataset contains features like glucose levels and other medical features, which can significantly affect the model performance.  

 




### Instructions for Execution :

- Step 1:

Make shore that your environment is capable of running Python  

`Go to the project folder using cd .../Project1-main and Run`


```bash
  pip install -r requirements.txt
```


- Step 2: 

Execute the CMD cd to go into the elasticnet folder module

```bash
  cd elasticnet
```

- Step 3:

To Execute the code Run  

```bash
  python __init__.py 
```

Which also runs test_predict() which can be used to test  different data using fit() and predict() for the models L1,L2, ElasticNet and KNN at same time.

- Step 4:

To run the test with k-fold cross validation you can find the code in ipynd file which splitting a dataset into a number of folds as needed (5 as default) and using each fold for training and validation in turn. Which can improve the MSE and R^2 (near to 1 for generated data) for ELastic net model. Which also gives you visualization better understanding apart from CV.

- Step 5:

To run the test without k-fold cross validation you can find the code in ipynd and __init__.py  file which runs the fit and predict methods for the X and Y provided.


### Code Description:
Here is the descrption for the implementation of every interface listed.
#### Related Methods: 


#### ScatterPlot() and run_models():
- `def ScatterPlot(y_true, y_pred, model_name, r2)`: This method generates scatter plot to visualize the relationship between true values(y_true) and predicted values (y_pred). Displays the R2 value on the plot and shows the best-fit line in red.
- `def run_models()`: In this method we make use of datasets (winequality-white.csv, diabetes.data.csv, dataset generated) to evaluate multiple regression models. Each dataset is cleaned and standardized (by feature scaling), with a bias term added for better model performance.
Linear Regression, Ridge regression (L2 Regularization), Lasso Regression (L1 regularization), Elastic Net Regression (Combination of L1 and L2 regularization), K-nearest Neighbors (KNN) regression are the models used in this method.

- Below are the steps for each dataset and model
	* The model is trained on the scaled features (X_scaled)
	* Predictions (y_pred) are generated
	* MSE and R2 metrics are calculated to assess model performance
		* `mse = (np.square(y - y_pred)).mean(axis=None)` 
             * MSE is used to calculate average squared difference between predicted and actual values. It measures the deviation from actual values.
		* `r2 = 1 - np.sum((y - y_pred) * 2) / np.sum((y - np.mean(y)) * 2) `
        
           * R2 indicates the goodness of fit for the model. 

#### Realted Methods:

* `def linear_loss(self, parameters, features, labels)`: Calculates the mean squared error loss for linear regression without regularization. It returns the loss value based on the current model parameters.
 * `def linear_gradient(self, parameters, features, labels)`: Computes the gradient of the loss function with respect to the parameters. This gradient is used for updating the model parameters during gradient descent.
* `def _build_tree(self, X, y, depth=0)`: Recursively builds the decision tree by finding the best feature and threshold to split the data.
* `def _get_best_split(self, X, y, num_features)`: Finds the best feature and threshold to split the data by evaluating different splits and calculating variance reduction.
* `def _split(self, X, y, feature_index, threshold)`: Divides the dataset into two subsets based on whether feature values are below or above the specified threshold for a given feature
* `def _calculate_variance_reduction(self, y, y_left, y_right)`: Computes the reduction in variance from a potential split by comparing the variance before and after split.
* `def elastic_net_loss(self, parameters, features, labels)`: Calculates the total loss as a combination of MSE and both L1 & L2 penalties.

* `def elastic_net_gradient(self, parameters, features, labels)`: Computes the gradient of the loss function, including the gradients for the MSE and regularization terms (L1 and L2)
* `def lasso_loss(self, parameters, features, labels)`: Computes the total loss for the Lasso model, which includes the linear regression loss (MSE) combined with an L1 regularization term that penalizes large coefficients to encourage  sparsity.
 * `def lasso_gradient(self, parameters, features, labels)`: Calculates the gradient of the Lasso loss function, which includes the gradient of the linear regression loss plus the gradient of the L1 regularization (the sign of the parameters).
* `def ridge_loss(self, parameters, features, labels)`: Computes the total loss for the Ridge model, which combines the linear regression loss (MSE) with an L2 regularization term that penalizes the square of the coefficients to reduce model complexity and prevent overfitting.
* `def ridge_gradient(self, parameters, features, labels)`: Calculates the gradient of the Ridge loss function, which consists of the gradient of the linear regression loss plus the gradient of the L2 regularization term (which is twice the parameter values).

Every funtion is called using super method and can be used without parameters as it contains default values which can be executes the Fit and Predict methods as per instrucuted and it is shown in `test_predict()`.


#### ElasticNet model implementaion using K-Fold Cross validation:

* `ElasticNet`:  ElasticNet class implements ElasticNet regression model from scratch along with K-Fold Cross Validation 

* `_init_()`: This function is the constructor function for the ElasticNet class and it initializes and assigns default arguments

* `fit(self, X, y)`: This method trains the model using gradient descent by adjusting the model coefficients based on input features (X) and target values (y). It adjusts the trade-off between feature selection and coefficient shrinkage by combining both L1 and L2 penalties during the optimization process

* `def predict(self, X)`: When new data (X) is given, this method uses the learned coefficients to generate predictions
* `def score(self, X, y)`: This method calculates the performance of the model by measuring two performance metrics – Mean Squared Error (MSE) and R2 (Coefficient of Determination).

* MSE measures the average squared difference between predicted and actual values. It measures the deviation from actual values.
* R2 indicates the goodness of fit for the model
* `def k_fold_cross_validation(model, X, y, n_splits=5)`: k_fold_cross_validation method splits the dataset into n_splits and iteratively trains and validates the model on different folds by calculating the value of MSE and R2.
It analyses the performance metrics using the value of MSE and R2 for each fold

Output:
```bash
Fold 1 - MSE: 12.569071852681008, R²: 0.9990183954891182
Fold 2 - MSE: 12.636289767434118, R²: 0.9990036622203452
Fold 3 - MSE: 12.406018075960594, R²: 0.9990534928957612
Fold 4 - MSE: 12.62360039258062, R²: 0.9990249782593933
Fold 5 - MSE: 12.416345013111266, R²: 0.9990464161127611
Average MSE over 5 folds: 12.530265020353522
Average R² over 5 folds: 0.9990293889954758
```

* Main execution block
This is the central part of the code from where the execution starts. read_csv() is used to read the datafile and later the input features (X) & target values (y) are extracted. 
`k_fold_cross_validation(model, X, y, n_splits=5)` is called to evaluate the model’s performance.





### Test Module:

Code:

```bash
def test_predict():
    print("\n ")
    print("----------------------------Test Data----------------------------")
    print("\n ")
    # Dictionary of model names and corresponding instances
    models = {
        "Lasso": LassoModel(),
        "Ridge": RidgeModel(),
        "KNN": KNNModel(),
        "ElasticNet": ElasticNetModel()
    }

    data = []

    current_dir = os.getcwd()

    file_path = os.path.join(current_dir, "tests", "test.csv")
    try:
        with open(file_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    
    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])
    
    for name, model in models.items():
        print("\n ")
        print(f"Testing {name}")
        print("\n ")
        model.fit(X, y)
        preds = model.predict(X)
        print(f"Predictions for {name}: {preds}")
        
        # Calculating Mean Squared Error (MSE)
        mse = np.mean((preds - y) ** 2)
        print(f"Mean Squared Error for {name}: {mse}")
        
        # Calculate R² score
        r2 = calculate_r2(y, preds)
        print(f"R² for {name}: {r2:.4f}")
        
    
        plt.figure(figsize=(8, 6))
        plt.scatter(y, preds, color='blue', label=f'{name} Predictions', alpha=0.7)
        plt.plot([min(y), max(y)], [min(y), max(y)], color='red', lw=2, label='Ideal Fit (y=x)')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{name} Model: Actual vs Predicted')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Optional: Add an assertion for the MSE threshold (e.g., below a certain value)
        assert mse < 22.0, f"High MSE for {name}: MSE = {mse}"

```


* `def calculate_r2(y_true, y_pred)`: This method calculates the R2 metric that measures how well the model’s predictions match the actual values. It is in the range of 0 to 1. Higher the value indicates better fit.
* `def test_predict()`: Reads the test.csv file located in the tests directory. It tests various models like lasso regression, ridge regression, K-nearest neighbors, elasticNet regression.
For testing diferent data, We can call the fit method by passing target values (X) and predicted values (y) from test_predict itself. This method trains the model using gradient descent by adjusting the model coefficients based on input features (X) and target values (y).
For each model it peforms model fitting, prediction, calculation of performance metrics (MSE and R2) and visualization (scatterplot)





