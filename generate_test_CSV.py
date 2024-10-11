import pandas as pd
from pydataset import data

# Load the 'Fair' dataset
fair_data = data('Fair')

# Save it as 'test.csv' inside the 'elasticnet/test/' directory
fair_data.to_csv('elasticnet/tests/test.csv', index=False)

print("test.csv has been generated and saved in elasticnet/test/")


