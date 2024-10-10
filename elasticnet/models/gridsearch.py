import numpy as np
from elasticnet.models.ElasticNet import ElasticNetLinearRegression
from sklearn.metrics import r2_score


def r2_score_manual(y_true, y_pred):

    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred)

    y_mean = np.mean(y_true)

    tss = np.sum((y_true - y_mean) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)

    r2 = 1 - (rss / tss)

    return r2


def grid_search_elastic_net(X_train, y_train, X_test, y_test, alpha_values, l1_ratio_values, learning_rate_values, max_iter_values):

    best_r2 = -np.inf
    best_params = {}

    for alpha in alpha_values:
        for l1_ratio in l1_ratio_values:
            for learning_rate in learning_rate_values:
                for max_iter in max_iter_values:

                    model = ElasticNetLinearRegression(alpha=alpha,
                                                       l1_ratio=l1_ratio,
                                                       learning_rate=learning_rate,
                                                       max_iter=max_iter)

                    model.fit(X_train, y_train)

                    predictions = model.predict(X_test)

                    r2 = r2_score(y_test, predictions)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_params = {
                            'alpha': alpha,
                            'l1_ratio': l1_ratio,
                            'learning_rate': learning_rate,
                            'max_iter': max_iter
                        }

    return best_params, best_r2
