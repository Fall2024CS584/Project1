# **ElasticNet Linear Regression Model**

## **Overview**

I have implemented a linear regression model with ElasticNet regularization. This model combines L1 (Lasso) and L2 (Ridge) penalties. The purpose of this model is to prevent overfitting and handle datasets with many correlated features.

I wrote this model from scratch in Python. I did not use pre-built machine learning libraries like `scikit-learn`. I only used `NumPy`, `Pandas`, and `Matplotlib` for numerical operations, data handling, and plotting.

## **What the Model Does**

ElasticNet regression models the relationship between a dependent variable and one or more independent variables. It applies both L1 and L2 penalties during training. The L1 penalty forces some coefficients to be zero. The L2 penalty helps handle multicollinearity by shrinking coefficients.

This model is useful when the dataset has many correlated features. It works when we want to regularize (shrink) coefficients. It works when we want to prevent overfitting.

## **How to Use the Model**

### **Parameters**

* **alpha**: Controls the strength of regularization. A higher value enforces more regularization. The default is 1.0.  
* **l1\_ratio**: Controls the balance between L1 and L2 regularization. A value of 0 makes it Ridge (L2) regression. A value of 1 makes it Lasso (L1) regression. The default is 0.5.  
* **learning\_rate**: Sets the step size for gradient descent. The default is 0.001.  
* **num\_iterations**: Defines the maximum number of gradient descent iterations. The default is 1000\.  
* **tolerance**: Sets the threshold for early stopping. The model stops if the cost function changes by less than this value. The default is 1e-4.

### **Model Methods**

#### **1\. fit(X, y)**

* **Inputs**:  
  * `X`: Feature matrix (2D numpy array).  
  * `y`: Target vector (1D numpy array).  
* **Function**: Trains the model using the feature matrix and target values.

#### **2\. predict(X)**

* **Inputs**:  
  * `X`: Feature matrix (2D numpy array).  
* **Function**: Returns predictions based on the trained model.

#### **3\. save\_model(filename)**

* **Inputs**:  
  * `filename`: Path where the model will be saved.  
* **Function**: Saves the trained model as a pickle file.

#### **4\. load\_model(filename)**

* **Inputs**:  
  * `filename`: Path where the model will be loaded from.  
* **Function**: Loads the trained model from a pickle file.

### **Steps to Run the Code**

**Preprocessing the Data**  
The dataset should contain only numerical features. I have written a preprocessing function to handle this. It removes non-numeric columns and handles missing values.  
Example:  
`data = pd.read_csv('prices.csv')  # Load dataset`  
`data = preprocess_data(data)      # Preprocess data`  
`X = data.iloc[:, :-1].values      # Feature matrix`  
`y = data.iloc[:, -1].values       # Target vector`

1. **Training the Model**  
   To train the model, use the `fit()` method:  
   `model = ElasticNetModel(alpha=0.5, l1_ratio=0.5, learning_rate=0.01, num_iterations=1000)`  
   `trained_model = model.fit(X_train, y_train)`

2. **Making Predictions**  
   After training, use the `predict()` method to get predictions:  
   `predictions = trained_model.predict(X_test)`

3. **Saving the Model**  
   To save the trained model:  
   `model.save_model('trained_model.pkl')`

4. **Loading the Model**  
   To load a saved model:  
   `loaded_model = ElasticNetModel()`

`loaded_model.load_model('trained_model.pkl')`

5. **Plotting the Cost Function**  
   The model tracks the cost function during training. You can plot the cost history to visualize convergence:  
   `plt.plot(range(len(model.cost_history)), model.cost_history)`  
   `plt.xlabel('Iterations')`  
   `plt.ylabel('Cost')`  
   `plt.title('Cost Function Convergence')`  
   `plt.show()`

## **How I Tested the Model**

I used `unittest` to test the model. I tested the model on synthetic data and a large real-world dataset.

### **Testing on Synthetic Data**

I generated synthetic data to check if the model works as expected. I also verified if the model's predictions match the true values. The model converged successfully.

* **Example Output**:  
  * Predictions: `[-25.292, 2.004, -5.137, -12.021, -2.639]`  
  * Actual values: `[-34.917, 2.187, -10.278, -16.017, -3.821]`  
  * Convergence time: `0.0575 seconds`  
  * Early stopping: Iteration 253

 ![image](https://github.com/user-attachments/assets/d4e776c5-8e17-4b15-a19c-bdcea3373613)
 ![image](https://github.com/user-attachments/assets/25ebb610-311d-4f5d-9899-0b352c189688)


### **Testing on Large Dataset (NYSE Prices Dataset)**

I tested the model with a large dataset (`prices.csv` from the New York Stock Exchange Kaggle dataset). The model took longer to converge but worked as expected.

* **Example Output**:  
  * Predictions: `[5214194.97, 4927342.16, 5835066.50, 5121165.88, 4952528.54]`  
  * Actual values: `[6795900, 4264000, 57652300, 1921400, 3300000]`  
  * Convergence time: `24.5056 seconds`  
  * Early stopping: Iteration 253
    
 ![image](https://github.com/user-attachments/assets/a69fb139-4e1a-462f-abca-0606dd23f0ff)
 ![image](https://github.com/user-attachments/assets/cc1f5103-cca3-4ae2-98ec-4e68a26e7dac)

## **Model Performance**

The model uses early stopping to improve performance. It stops training when the cost function changes very little. For small datasets, the model converges in a few iterations. For large datasets, the model takes longer to converge.

## **Limitations of the Model**

1. **Non-Numeric Data**: The model cannot process non-numeric features. Data must be numeric before training.  
2. **Imbalanced Datasets**: The model may struggle with datasets where one feature is much more common than others.  
3. **Scaling**: Data must be scaled. The model is sensitive to unscaled features.  
4. **Very Large Datasets**: The model may still take some time to train on very large datasets. Using stochastic gradient descent might improve performance further.

## **References**

Here are references I used for this project:

* **ElasticNet Regression**:  
  * [Elastic Net in Machine Learning: ElasticNet Explained](https://www.geeksforgeeks.org/implementation-of-elastic-net-regression-from-scratch/)  
* **L1 (Lasso) vs L2 (Ridge) Regularization**:  
  * [Regularization Techniques: Ridge and Lasso Regression](https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/)  
* **Gradient Descent**:  
  * [An Overview of Gradient Descent Optimization Algorithms](https://arxiv.org/abs/1609.04747)  
* **NY Stock Exchange Dataset**:  
  * [Kaggle \- New York Stock Exchange Dataset](https://www.kaggle.com/datasets/dgawlik/nyse?resource=download)

### **Additional Notes**

Parts of this code were debugged and error-corrected with assistance from a LLM, which helped ensure the functionality of the implementation.

[image1]: https://drive.google.com/file/d/1Xci-fCKlDNENSjlM-AOmlJEA9tFckqkz/view?usp=sharing
[image2]: https://drive.google.com/file/d/1Qj36gm50JqxXVTwQe8quOSCA8QGpLyEu/view?usp=sharing
[image3]: https://drive.google.com/file/d/1kDIVOYPooVbA8GVs_jD6_G9PMuTqvmRS/view?usp=sharing
[image4]: https://drive.google.com/file/d/15EVRianyn9JMjffi7KUdWnqAfNyG4H-c/view?usp=sharing

