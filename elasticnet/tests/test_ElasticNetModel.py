import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from elasticnet.models.ElasticNet import ElasticNetLinearRegression
from elasticnet.models.gridsearch import *
from elasticnet.models.checker import *


def test_predict():

    #! If you are going to use "pytest" enable it
    # file_path = "elasticnet/tests/small_test.csv"
    # df=pd.read_csv(file_path)
    # target ='y'

    #! Comment it out from here if you are going to use "pytest"
    file_path = input("Please enter the path to your dataset file: ")

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            print(
                "Unsupported file format. Please provide a CSV, Excel, JSON, or Parquet file.")
            return
    except FileNotFoundError:
        print("File not found. Please check the path and try again.")
        return

    print("\n" + "="*40)
    print("Dataset Preview:")
    print("="*40)
    print(df.head())

    target = input("Enter the target column name: ")

    #! Comment it out to here
    check_null(df)

    X, Y = XandY(df, target)

    np.random.seed(42)

    shuffled_indices = np.random.permutation(X.shape[0])

    train_size = int(0.8 * len(shuffled_indices))
    train_indices, test_indices = shuffled_indices[:
                                                   train_size], shuffled_indices[train_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = Y[train_indices], Y[test_indices]

    alpha_values = [0.1, 0.5, 1, 5, 10]
    l1_ratio_values = [0.1, 0.5, 0.7, 0.9]
    learning_rate_values = [0.01, 0.02, 0.05]
    max_iter_values = [500, 1000, 2000]

    best_params, best_r2 = grid_search_elastic_net(
        X_train, y_train, X_test, y_test, alpha_values, l1_ratio_values, learning_rate_values, max_iter_values
    )

    print("\n" + "="*40)
    print("Best Parameters from Grid Search")
    print("="*40)
    print(f"Alpha: {best_params['alpha']}")
    print(f"L1 Ratio: {best_params['l1_ratio']}")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"Max Iterations: {best_params['max_iter']}")

    print("\n" + "="*40)
    print(f"Best R² score: {best_r2:.4f}")
    print("="*40)

    final_model = ElasticNetLinearRegression(
        alpha=best_params['alpha'],
        l1_ratio=best_params['l1_ratio'],
        learning_rate=best_params['learning_rate'],
        max_iter=best_params['max_iter']
    )

    final_model.fit(X_train, y_train)
    final_predictions = final_model.predict(X_test)

    r2_score = final_model.r2_score_manual(y_test, final_predictions)

    mae_manual = final_model.mae_manual(y_test, final_predictions)
    rmse_manual = final_model.rmse_manual(y_test, final_predictions)

    print("\n" + "="*40)
    print("Final Model Evaluation")
    print("="*40)
    print(f"R² score: {r2_score:.4f}")
    print(f"Mean Absolute Error: {mae_manual:.4f}")
    print(f"Root Mean Squared Error: {rmse_manual:.4f}")
    print("="*40 + "\n")

    y_test = np.array(y_test).ravel()

    plt.figure(figsize=(8, 6))
    sns.kdeplot(y_test, color='blue', fill=True, label='Actual Values')
    sns.kdeplot(final_predictions, color='green',
                fill=True, label='Predicted Values')
    plt.title('Density Plot of Actual vs Predicted Values')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, final_predictions, color='blue',
                label='Predicted Values', alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
             color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction Error Plot')
    plt.legend()
    plt.grid(True)
    plt.show()


test_predict()
