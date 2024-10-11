### README: Video Game Sales Prediction Model

#### Project Title:
**Video Games Sales Prediction Using ElasticNet Regression**

#### Team Members:
1. Rithika Kavitha Suresh - A20564346
2. Ekta Shukla - A20567127
3. Roger Kewin Samson - A20563057
4. Jude Rosun - A20564339

---

### 1. Project Overview

This project aims to predict global sales for video games based on various features (e.g., year, region-specific sales). We employed data preprocessing, feature engineering, and a custom ElasticNet regression model to forecast global sales. The model helps predict sales figures and could be useful for video game publishers, analysts, and marketers in estimating future revenues.

### 2. Dataset Used
We used the **vgsales.csv** dataset containing information on video game titles, including sales across different regions and release years. 

- **Target Variable:** Global Sales
- **Features:** Year, NA_Sales, EU_Sales, JP_Sales, Other_Sales, etc.

### 3. Dependencies and Libraries
- **pandas** for data manipulation
- **numpy** for numerical operations
- **matplotlib** for visualizations
- **sklearn** for splitting the data into training and test sets (train_test_split)

#### Installation:
You can install these libraries using pip:
```bash
pip install pandas numpy matplotlib scikit-learn
```

### 4. Code Walkthrough and Purpose

#### 4.1 Importing Libraries and Loading Data
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('/path/to/vgsales.csv')
```
- **Purpose:** Import necessary libraries for handling data, visualizing results, and splitting the dataset for training and testing.

#### 4.2 Data Visualization (Global Sales vs Year)
```python
plt.scatter(df['Year'], df['Global_Sales'])
plt.show()
```
- **Purpose:** Plot the relationship between video game release year and global sales to visualize the raw data before any cleaning.

#### 4.3 Data Cleaning and Processing
```python
df['Year'].fillna(df['Year'].median(), inplace=True)
df['Publisher'].fillna('Unknown', inplace=True)
# Handling missing sales values
df.fillna(0, inplace=True)
df.dropna(subset=['Global_Sales'], inplace=True)
```
- **Purpose:** Handle missing data by filling missing values for years and publishers. We dropped rows where the target variable (Global Sales) was missing, as these rows are critical for training.

#### 4.4 Feature Engineering
```python
X = df[['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
y = df['Global_Sales']
```
- **Purpose:** Selected relevant features and the target variable. Features include year and regional sales.

#### 4.5 Splitting Data for Training and Testing
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
- **Purpose:** **sklearn’s `train_test_split`** is used to randomly split the dataset into 80% training and 20% testing data, ensuring that the model can generalize well. Inorder to avoid complication of spliting the data we took random_state as 0.

#### 4.6 ElasticNet Regression (Custom Implementation)
```python
class ElasticNetRegressionUpdated:
    # Model implementation with fit and predict methods
    def __init__(self, l1_ratio=0.5, alpha=0.01, max_iter=1000, tol=1e-4):
        pass
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass
```
- **Purpose:** Implement a custom ElasticNet regression algorithm, combining L1 and L2 penalties. This regularization technique is used to manage overfitting and improve prediction accuracy.
- 

#### 4.7 Model Training
```python
elastic_net_model_updated.fit(X_train, y_train)
```
- **Purpose:** Train the custom ElasticNet model on the training data.

#### 4.8 Predictions and Model Evaluation
```python
y_pred_updated = elastic_net_model_updated.predict(X_test)
mse_updated = np.mean((y_pred_updated - y_test) ** 2)
r_squared_updated = 1 - (np.sum((y_test - y_pred_updated) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
```
- **Purpose:** Evaluate the model using metrics such as **Mean Squared Error (MSE)** and **R-squared (R²)** to check prediction accuracy.

#### 4.9 Data Visualization (Predictions vs Actuals)
```python
plt.scatter(y_test, y_pred_updated)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()
```
- **Purpose:** Visualize the difference between actual and predicted values for further analysis.

---

### 5. Answering Key Questions

#### 1. **What does the model you have implemented do and when should it be used?**

The model implemented in this project is an **ElasticNet Regression** model, which combines **Lasso (L1)** and **Ridge (L2)** regression techniques. The primary purpose of the model is to predict **global sales of video games** based on features such as the year of release, sales in various regions (NA_Sales, EU_Sales, JP_Sales, Other_Sales), and other attributes in the dataset. 

ElasticNet is a linear regression model that includes penalties to avoid overfitting and handle datasets where there is **multicollinearity** between features, meaning some features are highly correlated with each other. In the case of video game sales data, regional sales figures may overlap in terms of contribution to global sales, making this model ideal because it manages correlated variables well.

- **When should this model be used?**
    - **Sales Forecasting:** When predicting future sales of video games based on historical data, ElasticNet's regularization helps avoid overfitting to past sales patterns.
    - **Feature Selection:** ElasticNet automatically reduces the impact of irrelevant features by setting some coefficients to zero, helping in situations where only a subset of the features is important.
    - **Handling Multicollinearity:** If the dataset contains correlated features (e.g., regional sales), the model is effective because it balances Lasso and Ridge penalties to minimize redundancy between features.

ElasticNet can be particularly useful for business forecasting and decision-making processes where both **prediction accuracy** and **interpretability** of the model are important.

---

#### 2. **How did you test your model to determine if it is working reasonably correctly?**

The model was tested using a well-established procedure to ensure its predictions are accurate and generalize well to new, unseen data.

- **Data Split for Training and Testing:** We used **`train_test_split`** from the `sklearn` library to randomly split the dataset into **80% training data** and **20% test data**. This allows the model to learn patterns from the training set and then be evaluated on the test set, which simulates real-world unseen data.
  
- **Evaluation Metrics:**
  - **Mean Squared Error (MSE):** We calculated the **MSE**, which measures the average squared difference between the actual and predicted values. A lower MSE indicates that the model's predictions are close to the true values.
  - **R-squared (R²) Score:** The **R² score** was used to evaluate how well the model captures the variance in the data. An R² score of 1 indicates perfect predictions, while a score of 0 means that the model is no better than predicting the mean. In this case, we obtained a reasonable R² score, indicating that the model is performing well in explaining the variation in global sales.

- **Visualization:** We also created **scatter plots** comparing actual vs. predicted sales values. A well-performing model should have points close to a straight line where the actual and predicted values match. By plotting the predictions and actuals, we were able to visually confirm that the model was making reasonable predictions.

These methods allowed us to ensure that the model was not only training effectively but also generalizing well to new data.

---

#### 3. **What parameters have you exposed to users of your implementation in order to tune performance?**

We exposed several key parameters in our ElasticNet model to allow users to fine-tune its performance. These parameters are particularly important for adjusting the behavior of the regularization and convergence during model training.

- **`l1_ratio`:** This parameter controls the balance between **L1 (Lasso)** and **L2 (Ridge)** regularization.
  - **0** corresponds to pure Ridge regression (L2 regularization).
  - **1** corresponds to pure Lasso regression (L1 regularization).
  - **In-between values** represent a mix of both. A value of `0.5` would use an equal mixture of Lasso and Ridge penalties. Users can tune this value to see if Lasso or Ridge is more effective for their data.

- **`alpha`:** This controls the overall strength of the regularization. Higher values of `alpha` increase the penalty applied to the model coefficients, shrinking them toward zero.
  - **Lower alpha values**: Allow for a less constrained model, potentially leading to overfitting.
  - **Higher alpha values**: Lead to more regularization, reducing overfitting but potentially oversimplifying the model.
  - Users can experiment with this parameter to find the balance between bias and variance.

- **`max_iter`:** This sets the maximum number of iterations the model will run to converge.
  - In practice, we set a default of 1000 iterations, but users can increase this if they observe the model is taking longer to converge or reduce it to speed up training time.
  
- **`tol`:** Tolerance for stopping criteria.
  - If the change in coefficients between iterations becomes smaller than `tol`, the model will stop optimizing. Reducing `tol` forces the model to find a better local minimum, while increasing it speeds up convergence but may sacrifice accuracy.

These parameters allow users to control the model's complexity, adjust for overfitting, and ensure good generalization to unseen data.

---

#### 4. **Are there specific inputs that your implementation has trouble with?**

The model works well on most of the data but could face challenges with the following types of inputs:

- **Outliers in Sales Data:** Extreme values or outliers in the regional sales or global sales data could distort the regression coefficients. ElasticNet helps mitigate this to some extent, but if there are too many outliers, the model's performance may degrade. Outliers can significantly affect prediction accuracy, as they can pull the regression line towards themselves, making the model less accurate for the majority of the data points.
  
- **Highly Sparse Data:** If a particular feature (e.g., sales in certain regions) has a lot of missing values or zeros, the model might struggle to learn meaningful patterns from this data. Sparse features contribute little information to the model, and ElasticNet may not be able to fully eliminate these features if `l1_ratio` is not appropriately tuned.

- **Non-linear Relationships:** While ElasticNet is useful for linear relationships between features and the target variable, it struggles when the relationship between regional sales and global sales is highly non-linear. Techniques like polynomial regression or neural networks might perform better in these cases.

- **Multicollinearity in Features:** Although ElasticNet can handle multicollinearity to some extent, if the collinearity between regional sales is too strong, it could still affect the stability of the model. ElasticNet is good at reducing this issue but cannot eliminate it entirely.

**Improvements with more time:**
- **Outlier Detection:** We could implement outlier detection techniques such as Z-scores or robust scaling methods to handle outliers more effectively.
- **Non-linear Transformations:** If given more time, we could explore adding non-linear transformations to the model (e.g., polynomial features) or using a more flexible model like **Random Forests** or **Gradient Boosting**.

---

#### **Why did we use `sklearn` for `train_test_split`?**

We used **`train_test_split`** from the **`sklearn`** library to handle the splitting of our dataset into training and testing subsets. The primary reason for using this function is its **simplicity** and **reliability** in creating randomized splits that ensure the model can generalize well.

- **Randomized Splitting:** `train_test_split` splits the data in a randomized way, ensuring that both the training and test sets contain a representative sample of the dataset. This randomness helps prevent overfitting to specific data points and ensures that the model is tested on unseen data.
  
- **Avoiding Data Leakage:** By using `train_test_split`, we avoid data leakage, which occurs when the testing data accidentally influences the training process. This ensures that the evaluation metrics are reliable and reflect the model's real-world performance.

- **Customizable Split Proportions:** It allows us to specify what portion of the data goes into the training set and what portion goes into the testing set. In this case, we used an 80-20 split (80% training and 20% testing), which is standard practice for machine learning problems.

**Why not use `train_test_split` for standardization?**
While `train_test_split` is useful for splitting data, we did not use it for standardization. **Standardization** refers to scaling the features so they have a mean of zero and a standard deviation of one and we knew the formula which can be easily implemented. For standardization, we would use **`StandardScaler`** from `sklearn`. Since the focus was on splitting the dataset, we used `train_test_split` only for this task. Standardization, if needed, would be done separately, especially after the split to avoid data leakage.

