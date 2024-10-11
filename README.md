# Project 1 

Put your README here. Answer the following questions.

* What does the model you have implemented do and when should it be used?
* How did you test your model to determine if it is working reasonably correctly?
* What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?


# <a name="_a97rca6syt7x"></a>**ElasticNet Linear Regression Model**
## <a name="_93e56w6gcy2k"></a>**Overview**
I have implemented a linear regression model with ElasticNet regularization. This model combines L1 (Lasso) and L2 (Ridge) penalties. The purpose of this model is to prevent overfitting and handle datasets with many correlated features.

I wrote this model from scratch in Python. I did not use pre-built machine learning libraries like scikit-learn. I only used NumPy, Pandas, and Matplotlib for numerical operations, data handling, and plotting.
## <a name="_2grgjzur0vd0"></a>**What the Model Does**
ElasticNet regression models the relationship between a dependent variable and one or more independent variables. It applies both L1 and L2 penalties during training. The L1 penalty forces some coefficients to be zero. The L2 penalty helps handle multicollinearity by shrinking coefficients.

This model is useful when the dataset has many correlated features. It works when we want to regularize (shrink) coefficients. It works when we want to prevent overfitting.
## <a name="_u63vfod1g327"></a>**How to Use the Model**
### <a name="_4lwnrojwd34l"></a>**Parameters**
- **alpha**: Controls the strength of regularization. A higher value enforces more regularization. The default is 1.0.
- **l1\_ratio**: Controls the balance between L1 and L2 regularization. A value of 0 makes it Ridge (L2) regression. A value of 1 makes it Lasso (L1) regression. The default is 0.5.
- **learning\_rate**: Sets the step size for gradient descent. The default is 0.001.
- **num\_iterations**: Defines the maximum number of gradient descent iterations. The default is 1000.
- **tolerance**: Sets the threshold for early stopping. The model stops if the cost function changes by less than this value. The default is 1e-4.
### <a name="_1ske51i9hhb1"></a>**Model Methods**
#### <a name="_5fs6zdiej287"></a>**1. fit(X, y)**
- **Inputs**:
  - X: Feature matrix (2D numpy array).
  - y: Target vector (1D numpy array).
- **Function**: Trains the model using the feature matrix and target values.
#### <a name="_mg7asils9hg"></a>**2. predict(X)**
- **Inputs**:
  - X: Feature matrix (2D numpy array).
- **Function**: Returns predictions based on the trained model.
#### <a name="_x3nmlgmfyu17"></a>**3. save\_model(filename)**
- **Inputs**:
  - filename: Path where the model will be saved.
- **Function**: Saves the trained model as a pickle file.
#### <a name="_t8tgrhhupdov"></a>**4. load\_model(filename)**
- **Inputs**:
  - filename: Path where the model will be loaded from.
- **Function**: Loads the trained model from a pickle file.
### <a name="_g8cjjpmf9uf8"></a>**Steps to Run the Code**
**Preprocessing the Data**

The dataset should contain only numerical features. I have written a preprocessing function to handle this. It removes non-numeric columns and handles missing values.

Example:

data = pd.read\_csv('prices.csv')  # Load dataset
data = preprocess\_data(data)      # Preprocess data
X = data.iloc[:, :-1].values      # Feature matrix
y = data.iloc[:, -1].values       # Target vector

1. **Training the Model**
   To train the model, use the fit() method:
   model = ElasticNetModel(alpha=0.5, l1\_ratio=0.5, learning\_rate=0.01, num\_iterations=1000)
   trained\_model = model.fit(X\_train, y\_train)

2. **Making Predictions**
   After training, use the predict() method to get predictions:
   predictions = trained\_model.predict(X\_test)

3. **Saving the Model**
   To save the trained model:
   model.save\_model('trained\_model.pkl')
   
4. **Loading the Model**
   To load a saved model:
   loaded\_model = ElasticNetModel()
   loaded\_model.load\_model('trained\_model.pkl')

5. **Plotting the Cost Function**
   The model tracks the cost function during training. You can plot the cost history to visualize convergence:
   plt.plot(range(len(model.cost\_history)), model.cost\_history)
   plt.xlabel('Iterations')
   plt.ylabel('Cost')
   plt.title('Cost Function Convergence')
   plt.show()

## <a name="_i0m88yd97j1l"></a>**How I Tested the Model**
I used unittest to test the model. I tested the model on synthetic data and a large real-world dataset.
### <a name="_35au61lh4gl8"></a>**Testing on Synthetic Data**
I generated synthetic data to check if the model works as expected. I also verified if the model's predictions match the true values. The model converged successfully.

- **Example Output**:
  - Predictions: [-25.292, 2.004, -5.137, -12.021, -2.639]
  - Actual values: [-34.917, 2.187, -10.278, -16.017, -3.821]
  - Convergence time: 0.0575 seconds
  - Early stopping: Iteration 253

### <a name="_c565p5y2duck"></a>**Testing on Large Dataset (NYSE Prices Dataset)**
I tested the model with a large dataset (prices.csv from the New York Stock Exchange Kaggle dataset). The model took longer to converge but worked as expected.

- **Example Output**:
  - Predictions: [5214194.97, 4927342.16, 5835066.50, 5121165.88, 4952528.54]
  - Actual values: [6795900, 4264000, 57652300, 1921400, 3300000]
  - Convergence time: 24.5056 seconds
  - Early stopping: Iteration 253
 
## <a name="_sovlnpqb9ecb"></a>**Model Performance**
The model uses early stopping to improve performance. It stops training when the cost function changes very little. For small datasets, the model converges in a few iterations. For large datasets, the model takes longer to converge.
## <a name="_x8sh9paovzu6"></a>**Limitations of the Model**
1. **Non-Numeric Data**: The model cannot process non-numeric features. Data must be numeric before training.
1. **Imbalanced Datasets**: The model may struggle with datasets where one feature is much more common than others.
1. **Scaling**: Data must be scaled. The model is sensitive to unscaled features.
1. **Very Large Datasets**: The model may still take some time to train on very large datasets. Using stochastic gradient descent might improve performance further.
