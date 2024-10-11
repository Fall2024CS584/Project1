import os
import csv
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from elasticnet.models.ElasticNet import ElasticNetModel

def grid_search_elastic_net(X_train, X_test, y_train, y_test):
    """
    Perform a grid search over regularization_strength, l1_ratio, and other parameters for the ElasticNet model.
    """
    # # Define a small grid for debugging This is for large dataset
    # regularization_strength_values = [0.01, 0.1, 0.5, 1.0, 5.0]
    # l1_ratio_values = [0.1, 0.2, 0.5, 0.7, 0.9]
    # learning_rate_values = [0.001, 0.01, 0.1]
    # max_iterations_values = [500, 1000, 2000]
    
    # This is for small dataset
    regularization_strength_values = [0.1, 0.5]
    l1_ratio_values = [0.1, 0.5]
    learning_rate_values = [0.01]
    max_iterations_values = [1000]
    

    best_r2 = -np.inf
    best_params = None

    # Loop over the grid of hyperparameters
    for regularization_strength in regularization_strength_values:
        for l1_ratio in l1_ratio_values:
            for learning_rate in learning_rate_values:
                for max_iterations in max_iterations_values:
                    print(f"Testing with regularization_strength={regularization_strength}, l1_ratio={l1_ratio}, "
                          f"learning_rate={learning_rate}, max_iterations={max_iterations}")
                    
                    try:
                        # Train ElasticNet model
                        model = ElasticNetModel(
                            regularization_strength=regularization_strength, 
                            l1_ratio=l1_ratio, 
                            max_iterations=max_iterations, 
                            learning_rate=learning_rate
                        )
                        results = model.fit(X_train, y_train)

                        # Evaluate model performance
                        r2 = results.r2_score(X_test, y_test)
                        print(f"R² Score: {r2}")

                        if r2 > best_r2:
                            best_r2 = r2
                            best_params = {
                                'regularization_strength': regularization_strength, 
                                'l1_ratio': l1_ratio, 
                                'learning_rate': learning_rate, 
                                'max_iterations': max_iterations
                            }
                    
                    except Exception as e:
                        print(f"Error during training with regularization_strength={regularization_strength}, l1_ratio={l1_ratio}: {e}")

    print("Best R² Score:", best_r2)
    print("Best Hyperparameters:", best_params)
    return best_params, best_r2

def test_predict():
    print("Current Working Directory:", os.getcwd())
    data = []
    csv_path = os.path.join(os.path.dirname(__file__), "data_long.csv")
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Open a text file to log results
    with open("results.txt", "w") as results_file:
        # Call the grid search function
        best_params, best_r2 = grid_search_elastic_net(X_train, X_test, y_train, y_test)

        # Log the best parameters and R² score
        results_file.write("Best R² Score: {}\n".format(best_r2))
        results_file.write("Best Hyperparameters: {}\n".format(best_params))

        # Optionally, train the final model with the best parameters
        if best_params is not None:
            model = ElasticNetModel(
                regularization_strength=best_params['regularization_strength'], 
                l1_ratio=best_params['l1_ratio'], 
                max_iterations=best_params['max_iterations'], 
                learning_rate=best_params['learning_rate']
            )
            results = model.fit(X_train, y_train)
            preds = results.predict(X_test)
            preds = np.array(preds).flatten()  # Ensure predictions are 1D
            y_test = np.array(y_test).flatten()  # Ensure actual values are 1D
            total_predictions = len(preds)
            print(f"Total number of predicted values: {total_predictions}")
            results_file.write(f"Total number of predicted values: {total_predictions}\n")

            # Check for NaN or Inf values in predictions
            if np.any(np.isnan(preds)) or np.any(np.isnan(y_test)):
                print("Warning: NaN values found in predictions or actual values.")
            if np.any(np.isinf(preds)) or np.any(np.isinf(y_test)):
                print("Warning: Inf values found in predictions or actual values.")

            # Log predictions, actual values, and their differences
            results_file.write("Predicted values: {}\n".format(preds))
            results_file.write("Actual values: {}\n".format(y_test))
            results_file.write("Differences: {}\n".format(np.abs(preds - y_test)))

            # Plot loss history and save the figure
            if hasattr(results, 'loss_history'):
                plt.figure(figsize=(10, 6))  # Set figure size for better visibility
                plt.plot(results.loss_history, color='blue')
                plt.title('Training Loss')
                plt.xlabel('Iterations')
                plt.ylabel('Train_Loss')
                plt.grid(True)
                plt.savefig("Training_Loss.png")
                plt.show()
                plt.close()  # Close the plot to avoid display
            else:
                print("No loss history available for plotting.")

            # Print summary and save to text file
            results_file.write("Model Summary:\n")
            results_file.write("Intercept: {}\n".format(results.intercept))
            results_file.write("Coefficients: {}\n".format(results.coefficients))
            results_file.write("Number of iterations: {}\n".format(len(results.loss_history)))
            results_file.write("Final loss: {}\n".format(results.loss_history[-1] if results.loss_history else "No loss recorded."))

            # Compute R² value and log it
            r2 = results.r2_score(X_test, y_test)
            results_file.write("Final R² Score: {}\n".format(r2))

            # Plot predictions vs actual values and save the figure
            plt.figure(figsize=(10, 6))  # Set figure size for better visibility
            plt.scatter(y_test, preds, color='blue', label='Predicted')
            plt.plot(y_test, y_test, color='red', label='Actual', linewidth=2)  # Ideal line
            plt.xlabel('Actual values')
            plt.ylabel('Predicted values')
            plt.title('Predicted vs Actual')
            plt.legend()
            plt.savefig("predictions_vs_actual.png")
            plt.show()
            plt.close()  # Close the plot

            # Plot residuals
            residuals = preds - y_test  # Calculate residuals
            plt.figure(figsize=(10, 6))  # Set figure size for better visibility
            plt.scatter(preds, residuals, color='blue', alpha=0.5)
            plt.axhline(0, color='red', linewidth=2, linestyle='--')  # Add a horizontal line at 0
            plt.title('Residuals Plot')
            plt.xlabel('Predicted values')
            plt.ylabel('Residuals')
            plt.grid(True)
            plt.savefig("residuals_plot.png")  # Save the residuals plot
            plt.show()
            plt.close()  # Close the plot

            # Log residuals to the file
            results_file.write("Residuals: {}\n".format(residuals))
            results_file.write("Mean Residual: {}\n".format(np.mean(residuals)))
            results_file.write("Standard Deviation of Residuals: {}\n".format(np.std(residuals)))

            # Optional assertion for validation
            tolerance = 10  # Example tolerance
            assert np.all(np.abs(preds - y_test) < tolerance), "Predictions do not match expected values within the tolerance."
        else:
            results_file.write("No best parameters found during grid search.\n")

if __name__ == "__main__":
    test_predict()
