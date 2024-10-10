from elasticnet.models.ElasticNet import ElasticNetModel
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data(filename):
    data = pd.read_csv(filename)
    y = data['y'].values.reshape(-1, 1)
    X = data.drop('y', axis=1).values
    return X, y

def plot(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.savefig('results.png')
    plt.close()


def main():
    X, y = load_data('data.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = ElasticNetModel(alpha=0.1, l1_ratio=0.5)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test).reshape(-1, 1)
    
    plot(y_test, y_pred_test)

if __name__ == "__main__":
    main()