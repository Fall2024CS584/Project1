# Project 1 

Put your README here. Answer the following questions.
Member: Yuxuan Qian
* __What does the model you have implemented do and when should it be used?__
  Elastic net regression
  
* __How did you test your model to determine if it is working reasonably correctly?__
  I first split data into two sets: the training set and the test set.  The training set was used to train the model to get each variable’s weight and model bias. Finally, I predicted Y based on the test set and calculated the r square to see how good the model was.  The visualization shows the plot of true Y vs. predicted Y
  
* __What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)__
  Parameters: lambda, alpha, max iteration, learning rate, tolerance
In code, the above parameters show: lamb, alp, max_iterate, rate, tol

Users need to set the initial lambda and alpha. The default max iteration, learning rate, and tolerance are respectively 1000, 0.01, 0.001

Usage examples:
from project1 import ElasticNetModel as ENM

model = ENM(100, 0.5, max_iterate = 1000, rate = 0.0001, tol = 0.001)

model.fit(x_train, y_train)

__OR__

from project1 import ElasticNetModel as ENM

model = ENM(100, 0.5)

model.fit(x_train, y_train)


* __Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?__
  It is hard to know what the values of lambda and alpha are. I had to try different values to know which one was the best. It is a fundamental problem. Another trouble was when I was trying a USA housing dataset containing 5000 values. The model always reached infinite weights, and it did not work whatever I changed the values of alpha or lambda. However, after changing the learning rate, it performed better though the r square is 0.49. I guess it was caused by the model did not normalize the features. If given more time, I will work to make the model standardize the features on a similar scale. 
