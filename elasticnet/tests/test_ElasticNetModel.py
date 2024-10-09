import csv
import numpy as np
import pandas as pd
from elasticnet.models.ElasticNet import ElasticNetModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def test_predict():
    model = ElasticNetModel()
    data = []

    # Correct path to the CSV file
    csv_file_path = "elasticnet/tests/small_test.csv"

    # Reading the CSV data into a Pandas DataFrame for easier manipulation
    df = pd.read_csv(csv_file_path)

    # Separating the features (X) and target (y)
    X = df.drop(columns=['y'])
    y = df['y']

    # Handling categorical columns in X using OneHotEncoder
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse=False, drop='first')
        X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), index=X.index)
        X_encoded.columns = encoder.get_feature_names_out(categorical_cols)

        # Drop original categorical columns and concatenate the encoded columns
        X = X.drop(columns=categorical_cols)
        X = pd.concat([X, X_encoded], axis=1)

    # Handling categorical target variable y using LabelEncoder
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Converting the DataFrame to NumPy array for model input
    X = X.values

    # Split the data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Fitting the model on training data
    results = model.fit(X_train, y_train)

    # Making predictions on the test data
    preds = results.predict(X_test)

    # If y was categorical, convert predictions back to categorical labels
    if y.dtype == 'object':
        preds = label_encoder.inverse_transform(np.round(preds).astype(int))

    # Calculate the Mean Squared Error and R-squared for the test set
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Print the MSE, R², coefficients, and intercept
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")
    print(f"Coefficients (beta): {results.coef_}")
    print(f"Intercept (beta0): {results.intercept_}")

    # Print the number of non-zero coefficients
    non_zero_coefs = np.sum(results.coef_ != 0)
    print(f"Number of non-zero coefficients: {non_zero_coefs}")

    # Dummy assertion, replace with actual evaluation
    assert isinstance(preds, np.ndarray), "Prediction is not an array"
