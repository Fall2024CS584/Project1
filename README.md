# Project 1 

Team Members
Rohit kumar Mamidi - A20541036
Venkata Siva Satya Pavan Ganesh Maddula - A20541032

Put your README here. Answer the following questions.

* What does the model you have implemented do and when should it be used?
*   The Elastic Net regression model combines both L1 (Lasso) and L2 (Ridge) regularization techniques to improve the performance of linear regression model and used     to reduce multi colliniarity . It does this by combining L1 and L2 penalties to promote sparsity and stabilize coefficient.
*   
* How did you test your model to determine if it is working reasonably correctly?
*   I have used RMSE and R2 as a metrics to evaulate the model.
*   
* What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
*   Alpha  Controls the overall strength of regularization. Higher values lead to more regularization.
    L1_ratio: Specifies the balance between L1 and L2 penalties:
    Learning rate: In iterative optimization algorithms, this controls how much to change the weights during training.
    Iterations: The maximum number of iterations for the optimization algorithm to converge.

* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
*   Elastic Net regression faces several challenges, including highly imbalanced datasets, severe multicollinearity, and the presence of missing values, which may        result in biased predictions and unreliable coefficient estimates. To enhance performance, it’s beneficial to incorporate preprocessing techniques such as            feature scaling and addressing missing data, as well as utilizing advanced optimization methods. Furthermore, conducting comprehensive hyperparameter tuning          through approaches like Grid Search or Random Search can significantly improve model accuracy. Implementing these strategies will help mitigate the model's           limitations and boost its overall effectiveness.
