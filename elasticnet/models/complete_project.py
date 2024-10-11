import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the vgsales.csv dataset
file_path = 'vgsales.csv'
df = pd.read_csv(file_path)

df.head()


# **2. Data visualization (scatter plot) - Global sales vs year(before cleaning)**

# Scatter plot of original data (before cleaning)
plt.figure(figsize=(10, 6))
plt.scatter(df['Year'], df['Global_Sales'], color='blue', label='Original Data')
plt.title('Global Sales vs Year (Before Cleaning)')
plt.xlabel('Year')
plt.ylabel('Global Sales (in millions)')
plt.legend()
plt.show()


# Data cleaning and processing
# 
# 
# *   Filling missing value
# 
# 

# Fill missing Year values with the median year
median_year = df['Year'].median()
df['Year'].fillna(median_year, inplace=True)

# Fill missing Publisher values with 'Unknown'
df['Publisher'].fillna('Unknown', inplace=True)

# Handle missing sales values
df['NA_Sales'].fillna(0, inplace=True)
df['EU_Sales'].fillna(0, inplace=True)
df['JP_Sales'].fillna(0, inplace=True)
df['Other_Sales'].fillna(0, inplace=True)


df.dropna(subset=['Global_Sales'], inplace=True)

df['Year'] = df['Year'].astype(int)

df.isnull().sum()


# Feature and Splitting the dataset for Training and 

X = df[['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].values
y = df['Global_Sales'].values

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# **Train-test and standardization**

# Standardizing the data
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std
X_train_scaled, mean, std = standardize(X_train)
X_test_scaled = (X_test - mean) / std


# **Elasticnet model implementation**


class ElasticNetRegressionUpdated:
    def __init__(self, l1_ratio=0.1, alpha=0.01, max_iter=1000, tol=1e-4, learning_rate=0.01):
        self.l1_ratio = l1_ratio  
        self.alpha = alpha
        self.max_iter = max_iter  # Maximum iterations for gradient descent
        self.tol = tol            
        self.learning_rate = learning_rate  

    def fit(self, X, y):
        m, n = X.shape
        self.coef_ = np.zeros(n)
        self.intercept_ = 0

        for _ in range(self.max_iter):
            y_pred = np.dot(X, self.coef_) + self.intercept_
            residuals = y_pred - y

            gradient_w = (2/m) * np.dot(X.T, residuals) + self.alpha * ((1 - self.l1_ratio) * self.coef_ + self.l1_ratio * np.sign(self.coef_))
            gradient_b = (2/m) * np.sum(residuals)

            self.coef_ -= self.learning_rate * gradient_w
            self.intercept_ -= self.learning_rate * gradient_b

            if np.linalg.norm(gradient_w) < self.tol:
                break

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_




# **ElasticNet Regression Model Initialization and Training**


# Initialize and train the updated ElasticNet regression model
elastic_net_model_updated = ElasticNetRegressionUpdated(l1_ratio=0.1, alpha=0.01, max_iter=1000, tol=1e-4, learning_rate=0.01)
elastic_net_model_updated.fit(X_train_scaled, y_train)


# **Predictions and Evaluation of Updated ElasticNet Model**




y_pred_updated = elastic_net_model_updated.predict(X_test_scaled)
predicted_df_updated = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_updated})
predicted_df_updated.to_csv('predicted_vgsales_updated.csv', index=False)
predicted_df_updated.head()



# **Model evaluation**


# Calculating Mean Squared Error (MSE) and R-squared (R2)
mse_updated = np.mean((y_pred_updated - y_test) ** 2)
r_squared_updated = 1 - (np.sum((y_test - y_pred_updated) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print(f'Mean Squared Error: {mse_updated}')
print(f'R-squared: {r_squared_updated}')



# **Data Visualizations**

# Scatter plot of predicted vs actual Global Sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_updated, color='purple', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Ideal Fit')
plt.title('Predicted vs Actual Global Sales (Updated Model)')
plt.xlabel('Actual Global Sales (in millions)')
plt.ylabel('Predicted Global Sales (in millions)')
plt.legend()
plt.show()



predicted_df_updated = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_updated})
predicted_df_updated.to_csv('predicted_vgsales_updated.csv', index=False)
predicted_df_updated.head()
