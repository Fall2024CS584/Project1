<h1>Student Name and A#</h1>
<p>Harlee Ramos A20528450</p>
<p>Andres Orozco A20528634</p>

<p>This document contains two parts. The information on how to run the program and the answers to the Project 1 questions.</p>

<h2>How to run the ElasticNet model:</h2>

<p>Instructions on how to use the ElasticNet Regularization Model: The <code>ElasticNetModel</code> is a custom implementation of ElasticNet regression, combining L1 (Lasso) and L2 (Ridge) regularization with gradient descent optimization. It’s suitable for datasets with high dimensionality and multicollinearity.</p>

<h3>Importing the Model</h3>
<p>In your Python environment, import the <code>ElasticNetModel</code> following the code and examples below:</p>

<pre><code>from elasticnet.models.ElasticNet import ElasticNetModel
</code></pre>

<h3>Training the Model with fit</h3>
<p>To train the model, use the fit method. It requires:</p>
<ul>
  <li><code>X</code>: A 2D NumPy array with shape [n_samples, n_features] representing the features.</li>
  <li><code>y</code>: A 1D NumPy array with shape [n_samples] representing the target values.</li>
</ul>

<p>Example:</p>

<pre><code>import numpy as np
from elasticnet.models.ElasticNet import ElasticNetModel

# Generate example data
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = 2 * X[:, 0] - 3 * X[:, 1] + 1.5 * X[:, 2] + np.random.normal(0, 0.1, 100)

# Initialize the model
model = ElasticNetModel(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-5, learning_rate=0.01)

# Fit the model
results = model.fit(X, y)
</code></pre>

<h3>Making Predictions with predict</h3>
<p>To predict with the fitted model, use the predict method:</p>
<ul>
  <li><code>X</code>: A 2D NumPy array with the same number of features as the training data.</li>
</ul>
<p>Example:</p>

<pre><code># Predict using the trained model
y_pred = results.predict(X)

import numpy as np
from elasticnet.models.ElasticNet import ElasticNetModel

# Generate sample data
X = np.random.rand(100, 3)
y = 4 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 0.1, 100)

# Initialize and fit the model
model = ElasticNetModel(alpha=0.05, l1_ratio=0.7, max_iter=1000, tol=1e-5, learning_rate=0.01)
results = model.fit(X, y)

# Make predictions
y_pred = results.predict(X)

print("Sample predictions:", y_pred[:5])
</code></pre>

<h3>Notes</h3>
<ul>
  <li>Ensure that <code>X</code> and <code>y</code> are formatted as NumPy arrays before using <code>fit</code>.</li>
  <li>The <code>predict</code> method can be used for any dataset with the same number of features as the training data.</li>
</ul>

<h2>Answer to the Project 1 questions:</h2>

<h3>Brief introduction</h3>
<p>The machine learning project 1 implements an ElasticNet regression model, a linear regression technique that incorporates L1 (Lasso) and L2 (Ridge) regularization penalties. ElasticNet addresses the limitations of Lasso and Ridge by combining them, making it suitable for high-dimensional data with potentially collinear features.</p>
<p>The model is optimized using gradient descent, an iterative method for minimizing functions by adjusting coefficients in the direction of the steepest descent of the gradient. ElasticNet’s objective function can be written as:</p>

<p>The Elastic Net regression aims to minimize the following objective function:</p>
<p>&beta;&#770; = arg min<sub>&beta;</sub> (1 / 2n) * &#124;&#124;y - X&beta;&#124;&#124;<sub>2</sub><sup>2</sup> + &alpha; * (1 - l1_ratio / 2) * &#124;&#124;&beta;&#124;</p>

<h4>Where:</h4>
<p>&beta;&#770; represents the estimated coefficients (or parameters) of the ElasticNet regression model that minimize the objective function.</p>
<ul>
  <li><p><code>arg min<sub>&beta;</sub></code> this denotes the operation of finding the value of (the coefficient vector) that minimizes the objective function.</p></li>
  <li><p>(1 / 2n) &#124;&#124;y - X&beta;&#124;&#124;<sub>2</sub><sup>2</sup> is the residual sum of squares (RSS), which measures how well the model fits the data.</p></li>
  <li><p><code>y</code> is the vector of observed target values.</p></li>
  <li><p><code>X&beta;</code> are the predicted values (based on the feature matrix X and coefficients).</p></li>
  <li><p>&#124;&#124;y - X&beta;&#124;&#124;<sub>2</sub><sup>2</sup> is the squared Euclidean distance (L2 Norm) between the true target y and the predicted values X&beta;, which represents the total error.</p></li>
  <li><p><sup>1</sup>&frasl;<sub>2n</sub> this part uses a 2n factor for scaling and simplifies the gradient calculation; n represents the number of observations/data points.</p></li>
  <li><p>&alpha; is the regularization strength. It balances the trade-off between minimizing the error (RSS) and shrinking the coefficients to reduce model complexity.</p></li>
  <li><p>(1 - l1_ratio) / 2 &#124;&#124;&beta;&#124;&#124;<sub>2</sub><sup>2</sup> is the Ridge penalty to control overfitting.</p></li>
  <li><p>&#124;&#124;&beta;&#124;&#124;<sub>2</sub><sup>2</sup> The L2 norm (sum of squares) of the coefficients, which penalizes large values of &beta; to prevent overfitting.</p></li>
  <li><p>1 - l1_ratio / <sub>2</sub> This controls the contribution of the L2 penalty, based on the L1 ratio parameter.</p></li>
  <li><p>l1_ratio &middot; &#124;&#124;&beta;&#124;&#124;<sub>1</sub> is the Lasso penalty that induces sparsity.</p></li>
  <li><p>&#124;&#124;&beta;&#124;&#124;<sub>1</sub> The L1 norm (sum of absolute values) of the coefficients &beta;, which encourages sparsity by shrinking some coefficients to zero.</p></li>
  <li><p>(l1_ratio) This controls the contribution of the L1 penalty in the regularization mix. A value of 1 means only Lasso, while a value of 0 means only Ridge.</p></li>
</ul>

<h3>Soft Thresholding and Proximal Operator Perspective</h3>
<p>&beta;&#770; = prox<sub>1</sub> arg min<sub>&beta;</sub> (1 / 2n) &#124;&#124;y - X&beta;&#124;&#124;<sub>2</sub><sup>2</sup> + (2 / 2) &#124;&#124;&beta;&#124;&#124;<sub>2</sub><sup>2</sup></p>
