# *Elasticnet model implementation*


class ElasticNetRegressionUpdated:
    def _init_(self, l1_ratio=0.1, alpha=0.01, max_iter=1000, tol=1e-4, learning_rate=0.01):
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




# *ElasticNet Regression Model Initialization and Training*


# Initialize and train the updated ElasticNet regression model
elastic_net_model_updated = ElasticNetRegressionUpdated(l1_ratio=0.1, alpha=0.01, max_iter=1000, tol=1e-4, learning_rate=0.01)
elastic_net_model_updated.fit(X_train_scaled, y_train)


# *Predictions and Evaluation of Updated ElasticNet Model*


y_pred_updated = elastic_net_model_updated.predict(X_test_scaled)
predicted_df_updated = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_updated})
predicted_df_updated.to_csv('predicted_vgsales_updated.csv', index=False)
predicted_df_updated.head()
