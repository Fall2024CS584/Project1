## Group Member(s)
ZIRUI OU A20516756

### What does the model do and when should it be used?

The ElasticNetModel is a type of linear regression that combines L1 and L2 regularization. It's ideal for predicting continuous outcomes, especially useful in datasets with irrelevant features or highly correlated features.

This model should be used when you suspect that your data contains irrelevant features or when the features are highly correlated. It's particularly useful in scenarios where you need a model that's robust against issues like multicollinearity (where independent variables are correlated) and when you want to prevent overfitting in your predictive model.

### How did you test your model to determine if it is working reasonably correctly?
The script tests the ElasticNet model by training it on a set of training data, making predictions on a separate test dataset, and then calculating the Mean Squared Error (MSE) between the predicted and actual values to assess accuracy. It ensures the MSE is below a threshold of 1 to verify the model's performance.


### What parameters are exposed to users to tune performance?

- **`lr` (Learning Rate):** Controls the update magnitude of model coefficients.
- **`n_iter` (Number of Iterations):** Determines how many times the model will process the entire dataset.
- **`l1_ratio` (L1 Ratio):** Balances between L1 and L2 regularization.
- **`alpha` (Regularization Strength):** Adjusts the overall strength of the regularization.

#### Basic Usage Example
```python

model = ElasticNetModel(lr=0.01, n_iter=1000, l1_ratio=0.5, alpha=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)

```

### Are there specific inputs that your implementation has trouble with?
Yes, the model struggles with non-numeric data, missing values, due to its basic implementation.

### Given more time, could these issues be worked around?
Yes, with more time, enhancements like automatic handling of non-numeric data and missing values, could be implemented to make the model more robust and efficient.


### Before you RUN:
1. please using `pip install numba numpy` to install numba and numpy before run it.
1. And make sure `test_data.csv` and `train_data.csv` are in the correct location, if not there, use one of the `Data_Generator` scripts to generate it according to the platform you are using..
2. Now you should ready to run the test program using `python elasticnet\tests\test_ElasticNetModel.py`.
