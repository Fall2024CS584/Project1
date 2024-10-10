import numpy as np

class ElasticNetModel():
    # set default max iteration = 1000, learning rate = 0.01, toleration = 0.001
    def __init__(self, lamb, alp, max_iterate = 1000, rate = 0.01, tol = 0.001):
        self.lamb = lamb
        self.alp = alp
        self.iter = max_iterate
        self.tol = tol
        self.rate = rate

    # prediction 
    def predict(self, X):
        self.pred = X.dot(self.W) + self.b
        return self.pred

    # gradient descent to update weight and bias
    def update_weights(self):
        # get y hat
        pred = self.predict(self.X)
        dW = np.zeros_like(self.W)

        # clacualte gradient for W and b
        # when alp == 0, it's lasso regression
        for j in range(self.m):
            if self.W[j] > 0:
                dW[j] = (-2*np.dot(self.X[: , j], self.Y - pred) + self.lamb*(1-self.alp) + 2*0.5*self.lamb*self.alp*self.W[j]) / self.n
            else:
                dW[j] = (-2*np.dot(self.X[: , j], self.Y - pred) - self.lamb*(1-self.alp) + 2*0.5*self.lamb*self.alp*self.W[j]) / self.n
        
        db = -2 * np.sum(self.Y - pred) / self.n

        # get new W and b
        self.W = self.W - self.rate*dW
        self.b = self.b - self.rate*db

        return self

    def fit(self, X, Y):
        # get number of the sample and the number of the features
        self.n, self.m = X.shape
        # X values
        self.X = X
        # y values
        self.Y = Y
        # set initial weights and bias
        self.W = np.zeros(self.m)
        self.b = 0

        for i in range(self.iter):
            self.old_W = self.W
            self.update_weights()

            # check convergence of coefficient to stop
            if np.max(np.absolute(self.W - self.old_W)) < self.tol*np.max(self.W):
                return self
            
        return self

