import numpy as np

class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if num_samples >= self.min_samples_split and depth < self.max_depth:
            best_split = self._get_best_split(X, y, num_features)
            if best_split['variance_reduction'] > 0:
                left_subtree = self._build_tree(best_split['X_left'], best_split['y_left'], depth + 1)
                right_subtree = self._build_tree(best_split['X_right'], best_split['y_right'], depth + 1)
                return {
                    'feature_index': best_split['feature_index'],
                    'threshold': best_split['threshold'],
                    'left': left_subtree,
                    'right': right_subtree
                }
        return np.mean(y)

    def _get_best_split(self, X, y, num_features):
        best_split = {}
        max_variance_reduction = -float('inf')
        
        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, feature_index, threshold)

                if len(y_left) > 0 and len(y_right) > 0:
                    variance_reduction = self._calculate_variance_reduction(y, y_left, y_right)

                    if variance_reduction > max_variance_reduction:
                        max_variance_reduction = variance_reduction
                        best_split = {
                            'feature_index': feature_index,
                            'threshold': threshold,
                            'X_left': X_left,
                            'y_left': y_left,
                            'X_right': X_right,
                            'y_right': y_right,
                            'variance_reduction': variance_reduction
                        }
        
        return best_split

    def _split(self, X, y, feature_index, threshold):
        X_left = X[X[:, feature_index] <= threshold]
        y_left = y[X[:, feature_index] <= threshold]
        X_right = X[X[:, feature_index] > threshold]
        y_right = y[X[:, feature_index] > threshold]
        return X_left, y_left, X_right, y_right

    def _calculate_variance_reduction(self, y, y_left, y_right):
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)
        reduction = np.var(y) - (weight_left * np.var(y_left) + weight_right * np.var(y_right))
        return reduction

    def predict(self, X):
        return np.array([self._predict_single_input(x, self.tree) for x in X])

    def _predict_single_input(self, x, tree):
        if isinstance(tree, dict):
            feature_index = tree['feature_index']
            threshold = tree['threshold']

            if x[feature_index] <= threshold:
                return self._predict_single_input(x, tree['left'])
            else:
                return self._predict_single_input(x, tree['right'])
        else:
            return tree  

