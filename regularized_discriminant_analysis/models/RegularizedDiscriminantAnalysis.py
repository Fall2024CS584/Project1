

class RDAModel():
    def __init__(self):
        self.regularization_param = regularization_param
        self.top_discriminant_ = None
        self.mean_class_1_ = None
        self.mean_class_2_ = None


    def fit(self, X, y):
        # Combine X and y into a DataFrame for easy manipulation
        dataset = pd.concat([X, y], axis=1)
        dataset.columns = list(X.columns) + ['selector']
        
        # Calculate class means
        class_means = dataset.groupby('selector').mean()

        # Calculate within-class scatter matrix
        class_labels = np.unique(y) 
        n_features = X.shape[1]  
        S_w = np.zeros((n_features, n_features))
        for label in class_labels:
            X_class = X[y == label]  # Use X (training data)
            mean_class = class_means.loc[label].values
            scatter_matrix_class = np.dot((X_class - mean_class).T, (X_class - mean_class))
            S_w += scatter_matrix_class

        # Calculate between-class scatter matrix
        global_mean = np.mean(X, axis=0)  # Use X for global mean
        S_b = np.zeros((n_features, n_features)) 
        for label in class_labels:
            N_k = X[y == label].shape[0]
            mean_class = class_means.loc[label].values  
            mean_diff = mean_class - global_mean
            mean_diff = np.atleast_2d(mean_diff).T  
            S_b += N_k * np.dot(mean_diff, mean_diff.T)

        # Regularization of within-class scatter matrix
        S_w_regularized = S_w + self.regularization_param * np.identity(S_w.shape[0])

        # Solve generalized eigenvalue problem
        eigvalues, eigvectors = np.linalg.eig(np.linalg.inv(S_w_regularized).dot(S_b))
        sorted_indices = np.argsort(eigvalues)[::-1]  
        eigvectors = eigvectors[:, sorted_indices]  

        # Save the top discriminant (eigenvector)
        self.top_discriminant_ = eigvectors[:, 0]

        # Project the training data onto the top discriminant
        X_projected = X.dot(self.top_discriminant_)

        # Calculate the means for the two classes in the projected space
        self.mean_class_1_ = X_projected[y == class_labels[0]].mean()
        self.mean_class_2_ = X_projected[y == class_labels[1]].mean()
        
    def predict(self, X):
        # Project the test data onto the discriminant
        X_projected = X.dot(self.top_discriminant_)

        # Classify based on the nearest mean
        def classify(x):
            return 1 if abs(x - self.mean_class_1_) < abs(x - self.mean_class_2_) else 2

        # Apply classification to each test data point
        y_pred = np.array([classify(x) for x in X_projected])
        return y_pred
        
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


