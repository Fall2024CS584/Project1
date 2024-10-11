import csv
import pandas as pd
import numpy

from regularized_discriminant_analysis.models.RegularizedDiscriminantAnalysis import RDAModel

def test_predict():

    #Loading the datset
    #Any dataset can be used 
    #I'm using liver disorders dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data"
    columns = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 'selector']
    dataset = pd.read_csv(url, names=columns)
    #Separate Features and Target Variable
    X=dataset.drop('selector', axis=1) #Features
    y=dataset['selector'] #Target
    #Performing Normalization/ Standardization on datasets
    #I am selecting Z- score Standardization(standard scaling): Transforms each feature in the dataset to have mean = 0 and standard deviation = 1
    X_standardized= (X-X.mean())/X.std()

    # Train-Test split
    train_size = int(0.8 * len(X_standardized))
    X_train = X_standardized[:train_size]
    X_test = X_standardized[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

# Initialize the RDA model
rda = RDAmodel(regularization_param=0.5)

# Train the model using X_train and y_train
rda.fit(X_train, y_train)

# Predict on test data
y_pred = rda.predict(X_test)

# Evaluate the model
accuracy = rda.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
