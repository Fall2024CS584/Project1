#### Authors
- Clara Aparicio Mendez (A20599326)
- Juan Cantarero Angulo (A20598593)
- Raquel Gimenez Pascual (A20599725)
- Carlota Ruiz de Conejo de la Sen (A20600262)
# Regularized Discriminant Analysis (RDA) Model

## Introduction

 - **Description**: The model developed is a classifer based on the Regularized Discriminant Analysis (RDA). The RDA algorithm is a compromise between the Linear Discriminant Analysis (LDA) and the Quadratic Discriminant Analysis (QDA) that introduces a regularization term to improve stability and performance by shrinking the covariance matrices. This flexibility allows RDA to smoothly transition between LDA and QDA, depending on the level of regularization applied. 
 
 - **Use**: RDA is particularly useful in multiclass classification, especifically when dealing with high-dimensional data, where many features are potentially correlated. RDA helps prevent overfitting and provides a balance between LDA's simple linear boundaries and QDA's quadratic ones, making it more adaptable to different data distributions.

## Model Testing

- **Testing methods**: To test the model, we have generated synthetic data testing with different possible dimentions. Each class is represented by a multivariate normal distribution with a defined number of samples. Additionally, a visualization function was implemented to observe the distribution of the generated data. This allows us to visually inspect the model's performance and the impact of the regularization parameters. Furthermore, we have corroborate the performance of our model implementing a confusion matrix, where we can visualize the results of the model. Thanks to this approach our model has an accuracy higher thatn 99%.

## Parameter for Tuning Performance

To control the balance between LDA and QDA, the parameter &alpha; must be adjusted:

- When **&alpha; = 0**, the model behaves like LDA, using a single shared covariance matrix across all classes. 
- When **&alpha; = 1**, the model behaves like QDA, with each class having its own distinct covariance matrix.

To better understand the regularization process used in the model, the following formula is applied:

$$
\hat{\Sigma}_k(\alpha) = \alpha \hat{\Sigma}_k + (1 - \alpha) \hat{\Sigma}
$$

Where:
<ul>
  <li>&#x3A3;<sub>k</sub> is the covariance matrix for class <em>k</em>.</li>
  <li>&#x3A3; is the pooled covariance matrix, shared across all classes.</li>
  </ul>  

## Structure

The project is divided into three main parts. 

- **Data Generation** : The function *generate_classifier_data()* is implemented for creating synthetic data with specific centroids and covariance matrices. This function allows users to test the RDA model by simulating various data distributions.

- **RDA Model**:

  - *Class RDAClassifier()*: 
  
    - **Fitting**: the method *fit()* estimates the class-specific means and covariances to obtain both, the shared covariance matrix and the specifics, afterwards it applies regularization. 

  - *Class RDAClassifierResult()*:

    - **Log-Likelihood**: the method *_log_loss()* calculates the log-likelihood for each data point belonging to a given class based on the regularized covariance matrices to quantify how likely each point is to belong to a given class. The lowest is the log-loss the higher is the probability of belonging to a class 

    - **Prediction**: the method *predict()* calculates the probability of each data point belonging to all classes based on the class-specific and regularized covariance matrices. It returns the class with the highest score for each point and is used for visualizing predictions in 2D. Additionally, the method *predict_LL()* estimates the log-loss for each point, providing a numerical measure of its likelihood for each class, which is then expressed as a probability.

    - **Visualization**: the method *viz_everything()* first, projects the data points onto the two most significant eigenvectors of the class-specific covariance matrices. Afterwards, this function provides a visual representation of how the model separates the classes. The colour given to the classified points is chosen according to their predicted class.


- **Test**: 

  - *Synthetic data*: The model has been tested using synthetic data generated with varying centroids and covariance matrices. Additionally, it has been evaluated across different dimensional settings, accommodating both N classes and N features to evaluate its performance under various conditions 

  - *Visualization*: The generated classifications are visualized to verify the correct behavior of the model reducing the dimensions of the features to 2D. 

  - *Confusion Matrix*: To evaluate the accuracy of the classification, we implemented a confusion matrix that clearly shows which data points have been correctly classified and where misclassifications occurred. 
  
## Basic Usage Examples

Quick example of how to generate data, fit the model, and visualize the classification:

1. **Generate Synthetic Data**: The following examples create synthetic data using specified centroids and covariance matrices to simulate different class distributions.


   ```python
    c_1=numpy.array([-1,1])
    c_2=numpy.array([2,1])
    c_3=numpy.array([0,-2.5])
    c_4=numpy.array([-3,-3])
    centroid_list = [c_1,c_2,c_3,c_4]
    sigma_1 = numpy.array([[1,1],[0,1]])
    sigma_2 = numpy.array([[0.7, 0.5], [0.5, 1.2]])  
    sigma_3 = numpy.array([[1.0, 0.0], [0.0, 1.0]])  
    sigma_4 = numpy.array([[1.1, 0.3], [0.3, 0.5]]) 
    sigma_list = [sigma_1,sigma_2,sigma_3,sigma_4]
    nsamples_1 = 500
    nsamples_2 = 500
    nsamples_3 = 500
    nsamples_4 =350
    nsamples_list = [nsamples_1,nsamples_2,nsamples_3,nsamples_4]
    xs, ys, n_classes = generate_classifier_data(centroid_list,sigma_list,nsamples_list, 73784)
    ```

2. **Fit the RDA Model**: Fit the model to the generated data, adjusting the regularization parameter 

    ```python 
    alpha = 0.7
    model = LDAClassifier()
    result = model.fit(xs, ys, n_classes, alpha)
    ```
3. **Visualize the classification**: Use the visualization function to see how the model separates the data into different classes:

    ```python
    x_p, predictions = result.viz_everything(xs)
    plt.scatter(x_p[:, 0], x_p[:, 1], c=predictions)
    plt.show()
    ```

## Improvements

**1. Incorporating &gamma; Regularization**:  

One potential improvement to the current implementation is the inclusion of the &gamma; parameter, which allows further regularization of the covariance matrices. 
  
The formula for &gamma;-based regularization is:

$$
\hat{\Sigma}(\gamma) = \gamma \hat{\Sigma} + (1 - \gamma) \hat{\sigma}^2 \mathbf{I}
$$
Where:
  - &#x3A3; is the covariance matrix (either class-specific or pooled).
  - &#x3C3;<sup>2</sup> **I** represents the scalar covariance matrix, where **I** is the identity matrix and &#x3C3;<sup>2</sup> is the variance.
  - &gamma; is the regularization parameter:
    - &gamma; = 1: full covariance matrix.
    - &gamma; = 0: scalar covariance matrix.

Adding this parameter gives more flexibility in managing the balance between bias and variance, allowing a smoother shift between complex covariance structures and simpler ones.

**2. Automate the generation of data**:  

An improvement to the model could be automating the data generation process by allowing users to simply specify the number of classes, features, and the range for the number of samples. The centroids and covariance matrices would then be randomly generated within those parameters instead of manually inserting them.