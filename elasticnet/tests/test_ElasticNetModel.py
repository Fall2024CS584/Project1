
# Importing libraries execpt ElastiNet

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# Data loading and preparation part

def load_data(file_path, date_col='DATE'):
    return pd.read_csv(file_path, parse_dates=[date_col])

credit_spread_data = load_data('data/credit_spread.csv')
gdp_data = load_data('data/GDP.csv')
unemployment_data = load_data('data/UNRATE.csv')
cpi_data = load_data('data/CPI.csv')
stocks_data = load_data('data/stocks.csv')
interest_rate_data = load_data('data/interest_rates.csv')


def preprocess_data(df, start_date='2021-01-02', end_date='2022-12-31'): #until 1997 max
  df['DATE'] = pd.to_datetime(df['DATE'])
  all_dates= pd.date_range(start=df['DATE'].min(), end=df['DATE'].max(), freq='D')
  df = df.set_index('DATE').reindex(all_dates)
  df = df.ffill()
  df = df.reset_index().rename(columns={'index': 'DATE'})
  df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
  return df

credit_spread_data = preprocess_data(credit_spread_data)
gdp_data = preprocess_data(gdp_data)
unemployment_data = preprocess_data(unemployment_data)
cpi_data = preprocess_data(cpi_data)
stocks_data = preprocess_data(stocks_data)
interest_rate_data = preprocess_data(interest_rate_data)


merged_data = credit_spread_data.merge(gdp_data, on='DATE', how='outer')
merged_data = merged_data.merge(unemployment_data, on='DATE', how='outer')
merged_data = merged_data.merge(cpi_data, on='DATE', how='outer')
merged_data = merged_data.merge(stocks_data, on='DATE', how='outer')
merged_data = merged_data.merge(interest_rate_data, on='DATE', how='outer')

merged_data.replace('.', np.nan, inplace=True)

merged_data = merged_data.apply(pd.to_numeric, errors='ignore')
print(merged_data.dtypes)


# Data preparation for the model

imputer = SimpleImputer(strategy='mean')
numeric_columns = merged_data.select_dtypes(include=['float64', 'int64']).columns
merged_data[numeric_columns] = imputer.fit_transform(merged_data[numeric_columns])

X = merged_data.drop(columns=['BAMLH0A0HYM2', 'DATE']) 
y = merged_data['BAMLH0A0HYM2']

poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Number of Nan ")
print(merged_data.isna().sum())

print(merged_data)

class ElasticNetModel:
    def __init__(self, alpha=0.001, l1_ratio=0.5, max_iter=1000, tol=1e-5):
        self.alpha = alpha 
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for i in range(self.max_iter):
            y_pred = self._predict(X)

            gradient_weight = -(1/n_samples) * X.T.dot(y - y_pred) + self.alpha * ((1 - self.l1_ratio) * 2 * self.weight + self.l1_ratio* np.sign(self.weight))
            gradient_bias= -(1/n_samples) * np.sum(y - y_pred)

            weight_old = self.weight.copy()
            self.weight -= self.alpha * gradient_weight
            self.bias -= self.alpha * gradient_bias

            if np.sum(np.abs(self.weight - weight_old))<self.tol:
                break

    def predict(self, X):
        return self._predict(X)

    def _predict(self, X):
        return X.dot(self.weight) + self.bias

# copying Sklearn parameters so its easier
elastic_net = ElasticNetModel(alpha=0.001, l1_ratio=0.5, max_iter=10000, tol=1e-5)
elastic_net.fit(X_train_scaled, y_train)

y_pred = elastic_net.predict(X_test_scaled)

print("y_pred:", y_pred)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Real values')
plt.ylabel('Predicted')
plt.title('Prediction against actual values')
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, color='green')
plt.title('Residuals distribution')
plt.xlabel('Residuals')
plt.show()

corr_matrix = merged_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation matrix")
plt.show()

