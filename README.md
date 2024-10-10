# Project 1 
Course: CS584 - Machine Learning
Instructor: Steve Avsec

## Team Members
1. Munish Patel - mpatel176@hawk.iit.edu (A20544034)
2. Jaya Karthik Muppinidi - jmuppinidi@hawk.iit.edu (A20551726)
3. Meghana Mahamkali - mmahamkali@hawk.iit.edu (A20564182)
4. Nirusha Mantralaya Ramesh - nmantralayaramesh@hawk.iit.edu (A20600814)

## Linear Regression with ElasticNet Regularization
### Project Overview
Linear regression with ElasticNet regularization (combination of L2 and L1 regularization)
This project implements a ElasticNet Model, which combines L1 and L2 penalties to enhance model generalization and prevent overfitting. The model, developed from scratch in Python using NumPy, optimizes its parameters via gradient descent.

### Usage
  ```bash
   # Fit and predict using the model
      model = ElasticNetModel(lambdas=0.1, l1_ratio=0.5, iterations=1000, learning_rate=0.001)
      results = model.fit(x_train_scaled, y_train)
      predictions = results.predict(x_test_scaled)

      predicted_categories = np.clip(np.round(predictions), 0, len(label_encoder.classes_) - 1).astype(int)
      # Converting numeric predictions back to job role labels
      predicted_job_roles = label_encoder.inverse_transform(predicted_categories)
      
      print("Numerical Predictions:", predictions)
      print("Predicted Job Roles:", predicted_job_roles)
   ```
### Explanation of the Model
1. What does the model you have implemented do and when should it be used?

  * The ElasticNetModel implemented is a type of regularized linear regression that combines both L1 and L2 regularization. 
  * L1 Regularization (Lasso) helps in feature selection by shrinking some coefficients to zero, which is beneficial in models with high dimensionality.
  * L2 Regularization (Ridge) tends to shrink coefficients evenly and helps in dealing with multicollinearity and model stability by keeping the coefficients small.
  * The main reason behind using ElasticNet is to build a model with least complexity while excelling in occasions where features seem to relate or when there are more variables than cases. When it is desirable to decrease the model’s complexity with regards to the features contributing to collinearity, then ElasticNet can prove effective.
  * ElasticNet is used when we suspect or know there is multicollinearity in your data, have a large number of features, some of which might be irrelevant, need a model that can perform feature selection to improve prediction accuracy.


2. How did you test your model to determine if it is working reasonably correctly?

  * We evaluated our model by training it on a dataset that predicts suggested job roles.
  * To verify the models ability to generalize, we have divided the data into training sets and testing sets.
  * Fit the model on the training data using results = model.fit(x_train_scaled, y_train).
  * Make predictions on the testing data using results.predict(x_test_scaled).
  * We tested tthe model in test.py using small_data.csv and also tested it in generate_regression_data.py where we generated random data and stored it in data.csv


3. What parameters have you exposed to users of your implementation in order to tune performance?

  * ```lambdas```: Controls the strength of the regularization. A higher value means more regularization.
  * ```l1_ratio```: Balances between L1 and L2 regularization.
  * ```iterations```: Determines the number of iterations in the gradient descent algorithm.
  * ```learning_rate```: Controls the step size at each iteration while moving toward a minimum of the loss function.
  * Example Usage for random generated data:
    ```bash
        model = ElasticNetModel(lambdas=1.0, l1_ratio=0.5, iterations=1000, learning_rate=0.01)
        results = model.fit(X_scaled, y)
    
        predictions = results.predict(X_scaled)
    
        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, alpha=0.5)
        plt.title('Comparison of Actual and Predicted Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # Diagonal line for reference
        plt.grid(True)
        plt.show()
    
        print("Predictions:", predictions)
        print("Actuals:", y)
        return predictions, y
    
        predictions, actuals = test_model_with_generated_data('data.csv')
    ```

4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

  * Non-linear Relationships, The ElasticNetModel, being a linear model, inherently assumes that the relationships between the predictors and the response variable are linear. This assumption limits its ability to model complex, non-linear interactions effectively.
  * High-dimensional Data, Although ElasticNet is designed to handle multicollinearity and can perform feature selection via L1 regularization, it might still struggle with very high-dimensional data (p >> n scenario), where the number of features far exceeds the number of observations.
  * Categorical Features Handling, we used binary encoding, number encoding, dummy variable encoding in the implementation of the project, as we had more number of categorical features than numerical features in our dataset.
  * Further regularization parameter tuning and potentially combining dimensionality reduction techniques like PCA (Principal Component Analysis) before applying ElasticNet could improve model performance.
