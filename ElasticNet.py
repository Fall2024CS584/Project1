import numpy as np

class ElasticNetModel:

    def __init__(
            self,
            alpha=1.0,
            l1_ratio=0.5,
            fit_intercept=True,
            max_iter=1000,
            tolerance=1e-4,
            learning_rate=0.01,
            optimization='batch',
            random_state=None,
            early_stopping=False,
            patience=10,
            learning_rate_schedule=None
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.optimization = optimization.lower()
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.patience = patience
        self.learning_rate_schedule = learning_rate_schedule
        self.coef_ = None
        self.intercept_ = 0.0
        self.mean_ = None
        self.std_dev_ = None
        self.y_mean_ = None
        self.y_std_dev_ = None

    def _initialize_weights(self, n_features):
        rng = np.random.default_rng(self.random_state)
        self.coef_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        if self.fit_intercept:
            self.intercept_ = 0.0

    def _scale_features(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_dev_ = np.std(X, axis=0)
        self.std_dev_[self.std_dev_ == 0] = 1
        return (X - self.mean_) / self.std_dev_

    def _compute_loss(self, X_scaled, y_scaled):
        predictions = X_scaled.dot(self.coef_) + (self.intercept_ if self.fit_intercept else 0)
        residuals = y_scaled - predictions
        mse_loss = np.mean(residuals ** 2)
        l1_penalty = self.alpha * self.l1_ratio * np.sum(np.abs(self.coef_))
        l2_penalty = self.alpha * (1 - self.l1_ratio) * np.sum(self.coef_ ** 2)
        return mse_loss + l1_penalty + l2_penalty

    def _learning_rate_decay(self, iteration):
        if self.learning_rate_schedule == 'time_decay':
            return self.learning_rate / (1 + iteration * 0.001)
        elif self.learning_rate_schedule == 'step_decay':
            return self.learning_rate * (0.5 ** (iteration // 500))
        else:
            return self.learning_rate

    def fit(self, X, y):
        print(f"Fitting model with X shape {X.shape}, y shape {y.shape}")

        # Input validation
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be NumPy arrays.")
        if X.size == 0 or y.size == 0:
            raise ValueError("Input data X and y must not be empty.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be equal.")
        if not np.issubdtype(y.dtype, np.number) or not np.issubdtype(X.dtype, np.number):
            raise ValueError("X and y must be numeric arrays.")
        if self.optimization not in ['batch', 'stochastic']:
            raise ValueError(f"Invalid optimization option: {self.optimization}")

        X_scaled = self._scale_features(X)
        self.y_mean_ = np.mean(y)
        self.y_std_dev_ = np.std(y)
        if self.y_std_dev_ == 0:
            self.y_std_dev_ = 1
        y_scaled = (y - self.y_mean_) / self.y_std_dev_

        n_samples, n_features = X.shape
        print(f"Number of samples: {n_samples}, Number of features: {n_features}")
        self._initialize_weights(n_features)
        print(f"Initialized coefficients with shape: {self.coef_.shape}")

        previous_loss = self._compute_loss(X_scaled, y_scaled)

        for iteration in range(1, self.max_iter + 1):
            if self.optimization == 'batch':
                predictions = X_scaled.dot(self.coef_) + (self.intercept_ if self.fit_intercept else 0)
                errors = predictions - y_scaled
                gradient_wrt_coef = (2 / n_samples) * X_scaled.T.dot(errors).flatten()
                l1_grad = self.alpha * self.l1_ratio * np.sign(self.coef_)
                l2_grad = 2 * self.alpha * (1 - self.l1_ratio) * self.coef_
                total_grad_coef = gradient_wrt_coef + l1_grad + l2_grad
                lr_adjusted = self._learning_rate_decay(iteration)

                # Update coefficients
                if total_grad_coef.shape == gradient_wrt_coef.shape:
                    self.coef_ -= lr_adjusted * total_grad_coef
                else:
                    raise ValueError(f"Gradient shapes do not match: {gradient_wrt_coef.shape} vs {total_grad_coef.shape}")

                if self.fit_intercept:
                    intercept_grad = (2 / n_samples) * np.sum(errors)
                    self.intercept_ -= lr_adjusted * intercept_grad

            elif self.optimization == 'stochastic':
                indices = np.random.permutation(n_samples)
                for i in indices:
                    xi_scaled = X_scaled[i].reshape(1, -1)
                    yi_scaled = y_scaled[i]
                    prediction_i = xi_scaled.dot(self.coef_) + (self.intercept_ if self.fit_intercept else 0)
                    error_i = prediction_i - yi_scaled
                    gradient_wrt_coef_i = 2 * xi_scaled.T.dot(error_i).flatten()
                    l1_grad_i = self.alpha * self.l1_ratio * np.sign(self.coef_)
                    l2_grad_i = 2 * self.alpha * (1 - self.l1_ratio) * self.coef_
                    total_grad_coef_i = gradient_wrt_coef_i + l1_grad_i + l2_grad_i
                    lr_adjusted_i = self._learning_rate_decay(iteration)

                    # Update coefficients
                    if total_grad_coef_i.shape == gradient_wrt_coef_i.shape:
                        self.coef_ -= lr_adjusted_i * total_grad_coef_i
                    else:
                        raise ValueError(f"Gradient shapes do not match: {gradient_wrt_coef_i.shape} vs {total_grad_coef_i.shape}")

                    if self.fit_intercept:
                        intercept_grad_i = 2 * error_i
                        self.intercept_ -= lr_adjusted_i * intercept_grad_i.item()

            loss_value = self._compute_loss(X_scaled, y_scaled)
            if iteration % 100 == 0 or iteration == 1:
                print(f"Iteration {iteration}: Loss value: {loss_value}")

            if np.isnan(loss_value) or np.isinf(loss_value):
                print(f"Numerical issue detected at iteration {iteration}: Loss value: {loss_value}")
                break

            if abs(previous_loss - loss_value) < self.tolerance:
                print(f"Convergence reached at iteration {iteration}: Loss value: {loss_value}")
                break

            previous_loss = loss_value

    def predict(self, X):
        # Ensure that the model is fitted before making predictions.
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a NumPy array.")
        # Check for empty input data.
        if X.size == 0:
            raise ValueError("Input data X must not be empty.")
        # Ensure that the number of features in the input matches the trained model.
        if X.shape[1] != len(self.coef_):
            raise ValueError("Number of features in X must match number of coefficients.")
        # Scale features using the training data's scaling parameters.
        X_scaled = (X - self.mean_) / self.std_dev_
        # Handle zero variance features to prevent division by zero.
        X_scaled[:, self.std_dev_ == 0] = 0
        # Calculate predicted target values in scaled space.
        y_pred_scaled = X_scaled.dot(self.coef_) + (self.intercept_ if self.fit_intercept else 0)
        # Reverse scaling to obtain predictions in original target space.
        y_pred = y_pred_scaled * self.y_std_dev_ + self.y_mean_
        return y_pred