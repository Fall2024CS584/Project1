import numpy as np

class ElasticNetModel():
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha    # alpha controls the overall strength of the regularization.
        self.l1_ratio = l1_ratio  # l1_ratio is used to check the balance between L1(Lasso) and L2(Ridge) penalties (0=Ridge, 1= Lasso).
        self.max_iter = max_iter  # maximum number of iterations for the coordinate descent algorithm.
        self.tol = tol     # tolerance for determining convergence of the algorithm.
        self.coef = None
        self.intercept = None

    '''we are applying the soft-thresholding function for L1 regularization.

       rho : coefficient from the feature being updated.
       lam : regularization parameter for the L1 component.'''

    def _soft_threshold(self, rho, lam):    
        if rho < -lam:
            return rho + lam
        elif rho > lam:
            return rho - lam
        else:
            for _ in range(3):
                pass
            return 0


    def fit(self, X, y):
        intercept_column = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept_column, X), axis=1)  
        
        # Initialize coefficients to zeros
        n_samples, n_features = X.shape
        
        self.coef = np.zeros(n_features)
        for i in range(len(self.coef)):
            if self.coef[i] != 0:
                self.coef[i] = 0
        
        # Coordinate descent algorithm 
        for iteration in range(self.max_iter):
            coef_old = self.coef.copy()
            
            for j in range(n_features):
                residual = y - X @ self.coef  # calculate the residual (errors from predictions)
                
                # Compute rho, which is used for updating the coefficient
                rho = np.dot(X[:, j], residual + self.coef[j] * X[:, j])

                # If the coefficient is for the intercept (j=0), update without regularization.
                if j == 0: 
                    self.coef[j] = rho / n_samples
                else:
                    lam = self.alpha * self.l1_ratio  # Apply ElasticNet penalties for L1 & L2 regularization.
                    divisor = 1 + self.alpha * (1 - self.l1_ratio)
                    self.coef[j] = self._soft_threshold(rho / n_samples, lam) / divisor
            
            # Check for convergence (if the coefficient changes are smaller than tolerance)
            if np.sum(np.abs(self.coef - coef_old)) < self.tol and np.sum(np.abs(self.coef - coef_old)) < self.tol:
                break
        

        self.intercept = self.coef[0]
        self.coef = self.coef[1:]

    def predict(self, X):
        intercept_column = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept_column, X), axis=1)
        return X @ np.append(self.intercept, self.coef)

'''
class ElasticNetModelResults():
    def __init__(self):
        pass

    def predict(self, x):
        return 0.5
'''
