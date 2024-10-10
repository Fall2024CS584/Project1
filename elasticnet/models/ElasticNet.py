import numpy as np

class ElasticNetModel:
    def __init__(self, alpha, l1_ratio, max_iter, tol=1e-8):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    def _soft_threshold(self, rho, l1_penalty):

        if rho < -l1_penalty:
            return rho + l1_penalty
        elif rho > l1_penalty:
            return rho - l1_penalty
        else:
            return 0

    def _compute_loss(self, X, y, coef_, intercept_):
        
        mse_loss = self._compute_mse(X, y, coef_, intercept_)
        penalty = self._compute_penalty(coef_)
        return mse_loss + penalty
    
    def _compute_mse(self, X, y, coef_, intercept_):
        
        y_pred = X.dot(coef_) + intercept_
        return np.mean((y - y_pred) ** 2)

    def _compute_penalty(self, coef_):
        
        l1_penalty = self.l1_ratio * np.sum(np.abs(coef_))
        l2_penalty = (1 - self.l1_ratio) * np.sum(coef_ ** 2)
        return self.alpha * (l1_penalty + l2_penalty)   


    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Normalize the features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / X_std

        # Initialize coefficients and intercept
        coef_ = np.zeros(n_features)
        intercept_ = 0
        loss_history = []

        for iteration in range(self.max_iter):
            # Update intercept
            intercept_ = np.mean(y - X.dot(coef_))

            for j in range(n_features):
                # Predict the target values with all features except the j-th feature
                residual = y - (X.dot(coef_) + intercept_) + coef_[j] * X[:, j]

                # Compute the optimal coefficient for the j-th feature
                rho = np.dot(X[:, j], residual) / n_samples
                l1_penalty = self.alpha * self.l1_ratio
                l2_penalty = self.alpha * (1 - self.l1_ratio)

                # Apply the soft thresholding to get the new value of coef_[j]
                coef_[j] = self._soft_threshold(rho, l1_penalty) / (1 + l2_penalty)

            # Compute the loss and check stopping condition
            loss = self._compute_loss(X, y, coef_, intercept_)
            loss_history.append(loss)

            # Check for convergence
            if iteration > 0 and np.abs(loss_history[-2] - loss_history[-1]) < self.tol:
                break

        return ElasticNetModelResults(coef_, intercept_, X_mean, X_std, loss_history)

class ElasticNetModelResults:
    def __init__(self, coef_, intercept_, X_mean, X_std, loss_history):

        self.coef_ = coef_
        self.intercept_ = intercept_
        self.X_mean = X_mean
        self.X_std = X_std
        self.loss_history = loss_history

    def predict(self, X):

        # Normalize the input data with the same scaling applied in fit
        X = (X - self.X_mean) / self.X_std
        return X.dot(self.coef_) + self.intercept_

    def plot_loss_history(self):

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(self.loss_history)), self.loss_history, color='b', label='Loss', alpha=0.7)
        plt.title("Training Loss Over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss Value")
        plt.grid(True)
        plt.legend()
        plt.show()

    def print_summary(self):
        
        print("Model Summary:")
        print("=" * 30)
        print(f"{'Intercept:':<20} {self.intercept_:.4f}")
        print(f"{'Coefficients:':<20} {', '.join(f'{coef:.4f}' for coef in self.coef_)}")
        print(f"{'Number of iterations:':<20} {len(self.loss_history)}")
        print(f"{'Final loss:':<20} {self.loss_history[-1]:.4f}" if self.loss_history else "No loss recorded.")
        print("=" * 30)
