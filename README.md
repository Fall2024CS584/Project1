# Project 1 

Put your README here. Answer the following questions.

* What does the model you have implemented do and when should it be used?
  The model that I implemented is a Regularized Linear Discriminant Analysis (LDA) classifier. This model is usually used for binary classification tasks where the main objective is to separate the data points into categories based on their features. This model first finds the best linear combination of features that separates the two classes. This method is best when there is at least some linear separability. The regularization is to prevent overfitting especially if the datasets are small or noisy.
  
* How did you test your model to determine if it is working reasonably correctly?
  To test the model we first split the dataset into training data (80%) and testing data (20%). This is done to evaluate the model’s performance. And to test whether the model is working reasonably correctly I used the accuracy metric. The accuracy is calculated as the percentage of correctly predicted labels in the test set.
  
* What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
  i. Regularization Parameter (λ): Helps fit the data and keep the model generalizable to new data. Higher value of λ increases regularization and prevents     overfitting. Lower value of λ allows model to capture more variance in data.
  ii. Number of features: We can choose which features to include and which not to include and experiment with different combinations. 
  iii. Train- test split: Users can also adjust the percentage of training and testing sets to explore the different model performances.

* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
  Yes there are specific inputs that my implementation has trouble with.
    i.	Since LDA is a linear classifier, it struggles with datasets that cannot be separated by a straight line or hyperplane. The non- linear patterns may not be captured effectively.
    ii.	LDA requires large datasets. Small Datasets like the dataset I chose leads to decrease in performance since it may not be able to compute means and covariances properly.
    iii.	The main reason for decrease in the accuracy is because one class significantly outweighs the other which makes the majority class more favorable.
  To solve this,  techniques like re-sampling and adjusting class weights can be done if given more time to increase the accuracy.
