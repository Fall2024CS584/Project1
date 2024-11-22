import numpy as np
import csv
from sklearn.preprocessing import StandardScaler  # To standardize the features

# Assuming the ElasticNetModel is defined as provided above

def test_predict():
    model = ElasticNetModel(lambdas=1.0, l1_ratio=0.5, iterations=1000, learning_rate=0.01)
    data = []
    
    # Load data from the CSV file
    with open("/content/small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert all values to float for consistency
            data.append({k: float(v) for k, v in row.items()})

    # Extract features and targets
    X = np.array([[v for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([datum['y'] for datum in data if 'y' in datum])
    
    # Normalize the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit the model
    results = model.fit(X_scaled, y)
    
    # Make predictions
    preds = results.predict(X_scaled)

    # Print predictions to verify outputs
    print("Predictions:", preds)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(y, preds, alpha=0.5, color='blue')  # Plot predictions vs actual values
    plt.title('Actual vs. Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line
    plt.grid(True)
    plt.show()

# Run the test function
test_predict()
