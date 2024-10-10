# Project 1 
August Gross - A20481734

## What is ElasticNet and When Should It Be Used?

The model implemented is a ElasticNet Linear Regression model, which combines L1 and L2 regularization.

The model should be used when you do not want to overfit linear data, and when you want to balance feature selection and regularization.

## Testing and Validation

The model was tested on data generated from generate_regression_data.py. The model was trained on the training data and tested on the test data. A plot of the predicted values vs the actual values was created to visualize the performance of the model.

## Model Parameters

The model has the following parameters that can be tuned:
- alpha: The strength of the regularization term. Default is 0.01.
- l1_ratio: The ratio of L1 regularization to L2 regularization. Default is 0.5.

Example usage:
```python
from elasticnet.models.ElasticNet import ElasticNetModel

model = ElasticNetModel(alpha=0.1, l1_ratio=0.5)
results = model.fit(X_train, y_train)

predictions = results.predict(X_test)
```

## Limitations and Future Improvements

The model has trouble with data that is not linear, which is fundamental to the model. The model also has a fixed learning rate and iterations, which could be improved upon.
