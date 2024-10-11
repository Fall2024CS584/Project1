import numpy as np

class ElasticNetModel:
    def __init__(self, regularization_strength, l1_ratio, max_iterations, tolerance=1e-6, learning_rate=0.01):
#  
#         Set up the ElasticNet regression model.

#         Parameters used in the model are:
#         regularization_strength: Regularization strength (λ or also called alpha)
#         l1_ratio: The balance between L1 and L2 ratios. It ranges from 0 to 1 where 0(pure Ridge) and 1(pure lasso).
#         max_iterations: Maximum number of iterations allowed for gradient descent process
#         tolerance: Threshold. criteria where it determines when to exit the process
#         learning_rate: Step size for updating coefficients during gradient descent process
     
        self.reg_strength = regularization_strength
        self.l1_ratio = l1_ratio
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate

    def loss_calculation(self, X, y, coefficients, intercept):
        """Compute the ElasticNet loss i.e; sum of MSE (Mean Squared Error),
                                            L1(Lasso),
                                            L2(Ridge) penalties"""
        predictions = X.dot(coefficients) + intercept
        squared_error_loss = np.mean((y - predictions) ** 2)
        l1_regularization = self.l1_ratio * np.sum(np.abs(coefficients))
        l2_regularization = (1 - self.l1_ratio) * np.sum(coefficients ** 2)
        return squared_error_loss + self.reg_strength * (l1_regularization + l2_regularization)

    def fit(self, X, y):
     
        # Train the model on the data by applying gradient descent
        # Parameters used in this method are:
        # X: Feature matrix (n_samples, n_features)
        # y: Target vector (n_samples,)
      
        n_samples, n_features = X.shape

        # Normalize the features
        feature_mean = np.mean(X, axis=0)
        feature_std = np.std(X, axis=0)
        X = (X - feature_mean) / feature_std

        # Initialize coefficients and intercept
        coefficients = np.zeros(n_features)
        intercept = 0
        loss_history = []

        for iteration in range(self.max_iterations):
            predictions = X.dot(coefficients) + intercept
            residuals = predictions - y

            # Compute gradient for intercept
            intercept_gradient = np.sum(residuals) / n_samples
            intercept -= self.learning_rate * intercept_gradient

            # Compute gradient for coefficients (ElasticNet penalty)
            coef_gradient = X.T.dot(residuals) / n_samples + \
                            self.reg_strength * (self.l1_ratio * np.sign(coefficients) +
                                                (1 - self.l1_ratio) * 2 * coefficients)

            # Update coefficients
            coefficients -= self.learning_rate * coef_gradient

            # Record the loss
            loss = self.loss_calculation(X, y, coefficients, intercept)
            loss_history.append(loss)

            # Stopping condition (based on gradient tolerance)
            if np.linalg.norm(coef_gradient) < self.tolerance:
                break

        # Return the fitted model and results encapsulated in ElasticNetModelResults
        return ElasticNetModelResults(coefficients, intercept, feature_mean, feature_std, loss_history)

class ElasticNetModelResults:
    def __init__(self, coefficients, intercept, feature_mean, feature_std, loss_history):
    
        # Wraps the outcomes of the ElasticNet model following the fitting process.

        # Parameters used in the method are:

        # coefficients: Coefficients obtained from fitting the model.
        # intercept: Intercept value determined during model fitting.
        # feature_mean: Average value of the features (utilized for normalization).
        # feature_std: Standard deviation of the features (utilized for normalization).
        # loss_history: Record of the loss values tracked throughout the training process.
       
        self.coefficients = coefficients
        self.intercept = intercept
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.loss_history = loss_history

    def predict(self, X):
       
        # Generate predicteds target values based on the provided input features
        
        # Parameters used in the method are:

        # X: Feature matrix for which predictions will be generated.
        # Returns:
        # predictions: The predicted target values.
    
        # Normalize the input data with the same scaling applied in fit
        X = (X - self.feature_mean) / self.feature_std
        return X.dot(self.coefficients) + self.intercept
    
    def r_squared(self, X, y_true):
        
        # Compute there  R-squared value for the model using the provided data.

        # Parameters used in the method are :

        # X: Feature matrix.
        # y_true: Actual target values.
        # Returns:

        # R² value: The calculated R-squared statistic.
        
        # Predict the values
        predictions = self.predict(X)
        
        # Total sum of squares (variance of the data)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        
        # Residual sum of squares (variance of the errors)
        ss_residual = np.sum((y_true - predictions) ** 2)
        
        # Compute R²
        r2 = 1 - (ss_residual / ss_total)
        return r2


    def display_output_summary(self):
        
        # Print a summary of the fitted model, including coefficients and intercept.
        
        print("Model Summary:")
        print(f"Intercept: {self.intercept}")
        print(f"Coefficients: {self.coefficients}")
        print(f"Number of iterations: {len(self.loss_history)}")
        print(f"Final loss: {self.loss_history[-1]}" if self.loss_history else "No loss. recorded.")
