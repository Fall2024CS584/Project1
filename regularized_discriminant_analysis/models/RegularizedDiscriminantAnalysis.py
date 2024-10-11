import matplotlib
import matplotlib.pyplot as plt
import numpy

matplotlib.rcParams['figure.figsize'] = (20, 10)

class RDAModel():

    def _init_(self):
        pass
    
    def fit(self, xs, ys, n_classes, alpha):
        self.n_k = [] # Number of samples for each class
        self.x = [] # Data points for each class
        self.mu = [] # Mean vector for each class
        self.pi = [] # Prior Probabilities
        cov = []
        cov_final = None # Pooled covariance matrix 
        self.sigma = []

        # Handle multiple classes
        for i in range(n_classes):
            # Number of data in a class
            self.n_k.append(numpy.where(ys == i)[0].shape[0])
            # Data from each class
            self.x.append(xs[numpy.where(ys == i)[0],:])
            # Get the mean of all data of each class
            self.mu.append(numpy.mean(self.x[i], axis=0)) 
            # Get the prior probabilities (pis) of each class
            self.pi.append(self.n_k[i] / numpy.sum(self.n_k))
            # Covariance matrix of each class
            cov.append(numpy.matmul((self.x[i] - self.mu[i]).T, self.x[i] - self.mu[i])/self.n_k[i])
            # Previous calculation to sigma
            if cov_final is None:
                cov_final = cov[i]
            else:
                cov_final += cov[i]
        
        # Compute pooled covariance matrix for regularization
        # (Shared covariance matrix = The same for each class)
        cov_final /= n_classes
        
        # Use regularization parameter alpha
        # (Regularized covariance matrixes = A different matrix for each class)
        for i in range(n_classes):
            self.sigma.append(alpha*cov[i] + (1 - alpha)*cov_final)
            # When alpha = 1 QDA (each class has its covariance matrix)
            # When alpha = 0 LDA (pooled covariace matrox)
            
        return RDAModelResults(self.pi, self.mu, self.sigma)

class RDAModelResults():
    def __init__(self, pi, mu, sigma):
        self.pi = pi
        self.mu = mu
        self.sigma = sigma

    def predict(self, xs):
        num_classes=len(self.mu)
        scores = [] # contains discriminant scores
        for i in range(num_classes):
            sigma_inv = numpy.linalg.inv(self.sigma[i]) 
            # distance = numpy.matmul(numpy.matmul((p - self.mu[i]).T, sigma_inv), (p - self.mu[i]))
            distance = numpy.einsum('ij,ij->i', (xs - self.mu[i]) @ sigma_inv, (xs - self.mu[i]))
            d_score = -0.5*distance + numpy.log(self.pi[i])-0.5*numpy.log(numpy.linalg.det(self.sigma[i]))
            scores.append(d_score)
        
        scores = numpy.array(scores)  
        predictions = numpy.argmax(scores, axis=0) 
        #predictions = predictions.reshape(xs.shape)  
        
        return predictions
    
    def predict_LL(self, x):
        LL = self._log_loss(x)
        exp_LL = numpy.exp(LL)
        probabilities = exp_LL / numpy.sum(exp_LL, axis=0)
        return probabilities
    
    def viz_everything(self, xs, ys):

        num_classes = len(self.mu)
        sigma_inv_x = []
        p =[]
        x_p =[]
        
        # Projection of xs to reduce data dimension into 2D (just for visualizing purposes)
        if xs.shape[0] > 2:
            for i in range(num_classes):
                sigma_inv_x.append(numpy.linalg.inv(self.sigma[i])) 
                p.append(numpy.linalg.eig(sigma_inv_x[i])[1][:,-2:])
                x_p.append(numpy.matmul(xs[numpy.where(ys == i)[0],:], p[i]))
        
            x_p = numpy.vstack(x_p)
            
        else:
            x_p=xs
            
        predictions = self.predict(xs)
        predictions = numpy.vstack(predictions)
        return x_p, predictions

    def _log_loss(self, x):

        k = len(self.mu)  #n of classes
        log_loss = numpy.zeros(k)

        for i in range(k):
            sigma_inv_i = numpy.linalg.inv(self.sigma[i])
            for j in range(i+1, k):
                sigma_inv_j = numpy.linalg.inv(self.sigma[j])
                sigma_inv_mui = numpy.matmul(sigma_inv_i, x - self.mu[i])
                sigma_inv_muj = numpy.matmul(sigma_inv_j, x - self.mu[j])

                first_term = 1/2*numpy.log(numpy.linalg.det(self.sigma[j])/numpy.linalg.det(self.sigma[i]))
                second_term = -1/2*numpy.matmul((x - self.mu[i]).T, sigma_inv_mui)
                third_term = 1/2*numpy.matmul((x - self.mu[j]).T, sigma_inv_muj)
                forth_term = numpy.log(self.pi[i]/self.pi[j])

                log_loss[i] = first_term + second_term + third_term + forth_term
        
        return log_loss
    
