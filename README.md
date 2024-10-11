# Submitted by:
- **Sana Samad**: A20543001
- **Naveed Mohiuddin**: A20540067
- **Owaiz Majoriya**: A20546104

---

# ElasticNet Linear Regression

## Project Overview
This project implements Linear Regression with ElasticNet Regularization, combining both L1 (Lasso) and L2 (Ridge) regularization to improve model generalization. The goal is to provide a robust machine learning model that prevents overfitting and adapts to datasets with multicollinearity or irrelevant features. The ElasticNet model has been built from the ground up using gradient descent for optimization, without relying on libraries like Scikit-learn or Statsmodels.

## Features
- **ElasticNet Regularization**: Applies both L1 and L2 penalties to the regression coefficients.
- **Custom Gradient Descent**: Minimizes the ElasticNet cost function using batch gradient descent.
- **Configurable Hyperparameters**: Users can easily adjust the weight of L1 (lambda1) and L2 (lambda2) regularization, along with other training settings.

## Setup

### Prerequisites
To run the code, ensure you have the following Python packages installed:
- Python 3.x
- NumPy: For efficient matrix operations and linear algebra.
- Pandas (optional): For dataset handling.

Install them via pip:

```bash
pip install numpy pandas
```

### How to Run
1. **Download the Notebook**: Clone the repository or download the notebook file (ElasticNet.ipynb) to your local machine:
    ```bash
    git clone <your_repo_url>
    cd <your_project_directory>
    ```
   
2. **Run the Notebook**: Open the notebook in Jupyter or any Python IDE with notebook support and run each cell sequentially:
    ```bash
    jupyter notebook ElasticNet.ipynb
    ```

3. **Input Data**:
   - The notebook uses a dataset (`output.csv`). If you use a custom dataset, ensure it is loaded into a pandas DataFrame with numeric features and target values.


4. **Train the Model**: The code will:
   - Load and preprocess the data.
   - Initialize and train the ElasticNet model using gradient descent.
   - Evaluate the model's performance.
   - Compares the true and predicted values
   - Plots the loss curvature.

### Example
```python
# Model training example:
model = ElasticNet(alpha=0.5, lambda1=0.1, lambda2=0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Parameters
You can customize the following parameters for model tuning:
- `alpha`: Weighting between L1 and L2 regularization.
- `lambda1`: Controls the strength of L1 regularization (Lasso).
- `lambda2`: Controls the strength of L2 regularization (Ridge).
- `learning_rate`: Learning rate for gradient descent.
- `max_iterations`: Maximum number of gradient descent iterations.

## Outputs
- **Training Metrics**: The notebook prints the cost function values over iterations, giving insight into convergence.
- **Model Predictions**: Compare predicted values with actual target values to evaluate model performance.

## Notes on Performance
The ElasticNet model performs well when:
- There's multicollinearity between predictors.
- Some features are irrelevant, making L1 regularization useful for feature selection.

### Limitations
The gradient descent method may be slow for large datasets. Improving the convergence with adaptive learning rates or alternative optimization techniques could be considered with more time.

## How to Run the Example:
### Load the Data:
Ensure your dataset (`output.csv`) is in the correct format: numerical data where all columns except the last are features, and the last column is the target.

### Train the Model:
The ElasticNet model is trained on your dataset using `fit(X, y)`.

### Performance Metrics:
After training, the model is evaluated using the `evaluate_model()` function, which prints:
- Mean Squared Error (MSE).
- R-squared (R²).

### Visualizations:
The `plot_results()` function generates two plots:
- A True vs Predicted values plot to see how well the model fits the data.
- A Loss over iterations plot to visualize model convergence.

## Conclusion:
This updated code provides both performance metrics (MSE and R²) and visualizations (True vs Predicted values and Loss history). These enhancements give more insight into how the ElasticNet model performs and help in debugging or improving the model.

---

# QUESTIONS:

1. **What does the model do and when should it be used?**
   The model implements ElasticNet Linear Regression, which combines L1 (Lasso) and L2 (Ridge) regularization. It is used to prevent overfitting in cases where data has multicollinearity (correlated features) or irrelevant features. The L1 part encourages sparse solutions (feature selection), while the L2 part stabilizes the regression for collinear features.

2. **How was the model tested?**
   The model was tested by:
   - Splitting data into training and test sets.
   - Evaluating the predictions against ground truth using mean squared error (MSE) and visual comparisons of predicted vs actual values.
   - Monitoring the convergence of the cost function during gradient descent.

3. **Exposed Parameters for Tuning Performance:**
   Users can adjust the following parameters:
   - `alpha`: Controls the balance between L1 and L2 regularization.
   - `lambda1`: L1 regularization strength (Lasso).
   - `lambda2`: L2 regularization strength (Ridge).
   - `learning_rate`: Controls the step size in gradient descent.
   - `max_iterations`: Limits the number of gradient descent iterations.

4. **Troublesome Inputs and Possible Solutions:**
   The model may struggle with:
   - Large datasets: Gradient descent can be slow without further optimization techniques.
   - Highly non-linear data: ElasticNet assumes linear relationships, so performance may degrade if the data is non-linear. Given more time, more efficient optimization techniques like stochastic gradient descent or adaptive learning rates could improve performance, or using kernel methods for non-linearity.
