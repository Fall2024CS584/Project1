# Project 1

- Group Members Detail:

Sai Pranay Yada (A20553636)
Kevan Dedania (A20522659)
Hemanth Vennelakanti (A20526563)
Kiran Velamati (A20525555)

- What does the model you have implemented do and when should it be used?
  We are using the Wine Dataset, which includes features like acidity, sugars, chlorides, density, pH, sulfates, and alcohol content. Based on these features, our model estimates the quality of wine. The model is trained to establish a relationship among the specified features to provide an accurate prediction of wine quality. Therefore, this is a regression model.

- How did you test your model to determine if it is working reasonably correctly?
  We evaluated the model using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) values. The results indicated reasonable performance for the model, confirming that it works effectively for predicting wine quality.

- What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
  We have exposed parameters such as lambda (the regularization strength), learning rate, and the l1_ratio to tune the performance of our model. By adjusting these hyperparameters, users can optimize the model according to their specific needs.

- Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
  Initially, we encountered challenges in finding a suitable dataset. Specifically, we found datasets that contained more categorical columns than numerical columns. This made it difficult to establish relationships between features and the target variable. Consequently, we focused on datasets that primarily contain numerical columns, as this helps in forming a clearer relationship with the target variable—in this case, the quality of the wine. Given more time, we could explore techniques to preprocess categorical data, but our current implementation is fundamentally designed for numerical inputs.
