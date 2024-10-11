# Project 1 

# CS584_Project1
Midterm project for CS 584

Rebecca Thomson
A20548618

CS584 – Machine Learning

Purpose:
Predictions using Regularized Linear Discriminant Analysis for Classification.  This program should be used for automatic gamma regulation of a Classification LDA model. 

Description:  
This program performs a Regularized Linear Discriminant Analysis (as described in Section 4.3.1 of Elements of Statistical Learning, 2nd Edition) from first principals for purposes of classification.

This program takes in the clean array of attributes and a dependent classifier variable. 
    -Data with NA or other missing values will not be accepted (an error check exists).
    -Data will be checked for sufficient datapoints.

The dependent classifier variable:
    -Can be 2 or more classes
    -Do not need to be balanced
    -Each class should be identified by a unique number
    -Class identifying numbers need not be sequential.
    -Data MUST be in list or DataFrame format.

The attributes:
    -All attributes should be numerical and sequential or one-hot encoded.
    -The weighted covariance matrix (sigma-hat) of the attributes must be invertible.
    -Data MUST be in list or DataFrame format.
    -To provide a linear ‘decision boundary’, there should be at least number of attributes plus one as number of classes, otherwise QDA should be used.  

Program Procedure:
When the fit(Y,X) is called:
1.	The program will automatically separate the classes for all calculations.  At this time, the program uses the ‘sys’ library to provide a soft exit if the data is insufficient in size.  The given data sets will be split into an 80/20 training/testing sets to tune the gamma penalty value.
2.	The program will then calculate a weighted sigma-hat covariance matrix from the training set.  This sigma-hat is the starting point of the regularization.
3.	Next, this sigma-hat will be used to find a regularized LDA with Sigma(gamma).  Since LDA requires all classes to use the same sigma-hat value, the gamma regularization of equation 4.14 of Elements of Statistical Learning, 2nd Edition was chosen.  The alpha equation 2.13 is for regulating QLA towards LDA.  The program automatically iterates through gamma values, saving the gamma value and error rate.  Each iteration will calculate the error rate on the testing set using the original model with the modified sigma-hat.  The program predicts the classification of each data-point in the testing set, compares it to the actual classification, and adds +1 to the error rate if it is wrong.  It does this by using equation 4.10 with each classes’ values, and then picking the class with the highest value.  This equation is a proportional probability, and argmax on all values will predict the class.  Once the program reaches a cutoff of number of errors, the process stops.  The first gamma with the lowest error on the test set is chosen. This could be gamma =1.0, the default.  This gamma is used to over-write the original sigma-hat value, so that all future predicted values from this model are calculated using this sigma-hat.

When confussion_matrix(Y_predict, Y_actual) command is called:
A series of confusion matrixes can be created by using the model and calling  model.confussion_matrix(Y_Predicted, Y_Actual). Each 2X2 matrix will be created and printed for each class.

When the predict(X) command is called:
1.	First, create your model with the training data.
2.	This command will return a list of the predicted values using the input of X, which is in the format of a list of lists of the attribute values.

Problems:
1.	Unfortunately, the program must be given data that produces and invertible sigma-hat.  Failing to produce an invertible sigma-hat causes an error.  My program does print out a line summarizing the problem, but it cannot work around it.
2.	The program currently needs input dependent values to be in list format.  If I had time, I would produce a series of if-then statements to automatically convert all input into the working format.  I did provide a working conversion for attributes from DataFrame to list, but if the dependent variable input is a DataFrame, the conversion is very unreliable, sometimes adding extra brackets.
3.	The gamma chosen can be gamma=1.0 if the set is easily fit without overfitting problems. 

Global variables:
These are the global variables of each model that can be called for examination:

cov_list	              List of individual class covariance matrixes
mu_list	                List of individual class means for each attribute
pi_list	                List of pi values (# per class/totalN)
sigma_hat	              Sigma value for model, automatically tunned
gamma	                  The gamma value with the lowest error on the test set (float)
X_by_class	            A list of lists of each datapoint separated by class
totalN	                Total training datapoints in input model (int)
total_i_per_class	      # Datapoints per class (list)
classes_given	          List of all possible classes by identifying number in input data
classes_given_no	      Total number of classes (int)
attributes_no	          Total number of independent attributes (int)
sigma_hat_unmodified	  Unmodified value of sigma, for checking.
