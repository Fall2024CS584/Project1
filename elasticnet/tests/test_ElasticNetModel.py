import os
import csv
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from elasticnet.models.ElasticNet import ElasticNetModel
from datetime import datetime

# This function performs a grid search to find the best hyperparametrs
# like regularization strength, l1_ratio, learning_rate, and max_iterations
def ml_grid_search(X_train, X_test, y_train, y_test):
    
    #  This is for large grid parameter variations
    # regularization_strength_values = [0.01, 0.1, 0.5, 1.0, 5.0]
    # l1_ratio_values = [0.1, 0.2, 0.5, 0.7, 0.9]
    # learning_rate_values = [0.001, 0.01, 0.1]
    # max_iterations_values = [500, 1000, 2000]
    
    # This is for small grid parameter variations
    regularization_strength_values = [0.1, 0.5]
    l1_ratio_values = [0.1, 0.5]
    learning_rate_values = [0.01]
    max_iterations_values = [1000]
    

    best_r2 = -np.inf  # Start with a very low r2 value
    best_params = None  # To store the best parameters found

    # Iterate over all combinations of hyperparametr values
    for regularization_strength in regularization_strength_values:
        for l1_ratio in l1_ratio_values:
            for learning_rate in learning_rate_values:
                for max_iterations in max_iterations_values:
                    # Print the current set of hyperparameters being tested
                    print(f"Test with regularization_strength={regularization_strength}, l1_ratio={l1_ratio}, "
                          f"learning_rate={learning_rate}, max_iterations={max_iterations}")
                    
                    try:
                        # Initialize the ElasticNetModel with current parameters
                        model = ElasticNetModel(
                            regularization_strength=regularization_strength, 
                            l1_ratio=l1_ratio, 
                            max_iterations=max_iterations, 
                            learning_rate=learning_rate
                        )
                        results = model.fit(X_train, y_train)  # Fit the model to training data
                        r2 = results.r_squared(X_test, y_test)  # Compute the R² score
                        print(f"R² Score: {r2}")

                        # If this model has a better R² score, save its parameters
                        if r2 > best_r2:
                            best_r2 = r2
                            best_params = {
                                'regularization_strength': regularization_strength, 
                                'l1_ratio': l1_ratio, 
                                'learning_rate': learning_rate, 
                                'max_iterations': max_iterations
                            }
                    
                    except Exception as e:
                        # Print an error if the model fails to train with the current parameters
                        print(f"Error while training along with regularization_strength={regularization_strength}; l1_ratio={l1_ratio}: {e}")

    # Print the best R² score and corresponding hyperparameters
    print("Best R_square Score value:", best_r2)
    print("Best Hyperparameters here:", best_params)
    return best_params, best_r2  # Return the best parameters and the R² score

# Main function for testing predictions with the best model
def test_predict():
    print("Current Working Dir:", os.getcwd())  # Print the current directory
    data = []
    csv_path = os.path.join(os.path.dirname(__file__), "data_long.csv")  # Path to the CSV file
    
    # Read data from the CSV file
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Prepare feature matrix X and target vector y
    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get current timestamp
    results_file_path = f"results_{timestamp}.txt"  # File to store results
    
    # Open the results file to write the output
    with open(results_file_path, "w") as results_file:
        # Run grid search to find the best parameters
        best_params, best_r2 = ml_grid_search(X_train, X_test, y_train, y_test)

        results_file.write("Best R_square Score value: {}\n".format(best_r2))
        results_file.write("Best Hyperparameters here: {}\n".format(best_params))

        # If best parameters are found, retrain the model with them
        if best_params is not None:
            model = ElasticNetModel(
                regularization_strength=best_params['regularization_strength'], 
                l1_ratio=best_params['l1_ratio'], 
                max_iterations=best_params['max_iterations'], 
                learning_rate=best_params['learning_rate']
            )
            results = model.fit(X_train, y_train)  # Train the model with best params
            preds = results.predict(X_test)  # Make predictions on test data
            preds = np.array(preds).flatten()  # Flatten the predictions array
            y_test = np.array(y_test).flatten()  # Flatten the true values array
            total_predictions = len(preds)  # Count of total predictions made
            print(f"Total no of predicted values: {total_predictions}")
            results_file.write(f"Total no of predicted values: {total_predictions}\n")

            # Check for NaN or infinite values in predictions or true values
            if np.any(np.isnan(preds)) or np.any(np.isnan(y_test)):
                print("Warning: NaN values found in the predictions or the actual values")
            if np.any(np.isinf(preds)) or np.any(np.isinf(y_test)):
                print("Warning: Inf values found in the predictions or the actual values.")

            # Write predictions, actual values, and their differences
            results_file.write("Predicted values: {}\n".format(preds))
            results_file.write("Actual values: {}\n".format(y_test))
            results_file.write("Differences: {}\n".format(np.abs(preds - y_test)))

            # Plot the training loss if available
            if hasattr(results, 'loss_history'):
                plot.figure(figsize=(10, 6))
                plot.plot(results.loss_history, color='blue')
                plot.title('Training Loss')  # Set the title of the plot
                plot.xlabel('Iterations')  # X-axis label
                plot.ylabel('Train_Loss')  # Y-axis label
                plot.grid(True)
                plot.savefig("Training_Loss.png")  # Save the plot as an image
                plot.show()  # Display the plot
                plot.close()
            else:
                print("No loss history available for plotting the graph")

            # Write model summary: intercept, coefficients, etc.
            results_file.write("Model Summary:\n")
            results_file.write("Intercept: {}\n".format(results.intercept))
            results_file.write("Coefficients: {}\n".format(results.coefficients))
            results_file.write("Number of iterations: {}\n".format(len(results.loss_history)))
            results_file.write("Final loss: {}\n".format(results.loss_history[-1] if results.loss_history else "No loss recorded."))

            # Compute the final R² score on the test data
            r2 = results.r_squared(X_test, y_test)
            results_file.write("Final R_square Score: {}\n".format(r2))

            # Plot predicted vs actual values
            plot.figure(figsize=(10, 6))
            plot.scatter(y_test, preds, color='blue', label='Predicted')
            plot.plot(y_test, y_test, color='red', label='Actual', linewidth=2)
            plot.xlabel('Actual values')  # Label the X-axis
            plot.ylabel('Predicted values')  # Label the Y-axis
            plot.title('Predicted vs Actual')  # Title of the plot
            plot.legend()
            plot.savefig("predictions_vs_actual.png")  # Save the plot as an image
            plot.show()
            plot.close()

            # Plot the residuals (difference between predicted and actual)
            residuals = preds - y_test
            plot.figure(figsize=(10, 6))
            plot.scatter(preds, residuals, color='blue', alpha=0.5)
            plot.axhline(0, color='red', linewidth=2, linestyle='--')  # Line at y=0
            plot.title('Residuals Plot')  # Title for the residuals plot
            plot.xlabel('Predicted values')  # X-axis label
            plot.ylabel('Residuals')  # Y-axis label
            plot.grid(True)
            plot.savefig("residuals_plot.png")  # Save the residuals plot
            plot.show()
            plot.close()

            # Write residuals and their statistics to the results file
            results_file.write("Residuals: {}\n".format(residuals))
            results_file.write("Mean Residual: {}\n".format(np.mean(residuals)))
            results_file.write("Standard Deviation of Residuals: {}\n".format(np.std(residuals)))

            # Tolerance for differences between predicted and actual values
            tolerance = 10
            assert np.all(np.abs(preds - y_test) < tolerance), "Predictions don't match with the expected values /"
        else:
            results_file.write("No best parameters -during grid search.\n")

# Run the test_predict function when the script is executed
if __name__ == "__main__":
    test_predict()
