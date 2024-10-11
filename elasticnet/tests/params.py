# All User parameter can be adjusted from here.

"""
 Dataset configuration
"""

# User defined params for data import setting
data_file_params = {
    "file_name": "small_test.csv",  # File path to read
    "test_ratio": 0.3,  # Portion of the test set from total sample.
}

# User defined params for synthetic data creation setting
data_syn_params = {
    "n_samples": 100,  # Number of samples to create
    "n_features": 3,  # Number of features of the data
    "weights": None,  # Predetermined weights - If set as None, it creates the new one
    "bias": None,  # Predetermined bias - If set as None, it creates the new one
    "noise_std": 0.01,  # Noise scale to check durability of the model
    "random_state": 42,  # Random seed
    "test_ratio": 0.3,  # Portion of the test set from total sample.
}

# User defined params for multi-collinear data creation setting
data_mul_params = {
    "n_samples": 100,  # Number of samples to create
    "n_features": 3,  # Number of features of the data
    "weights": None,  # Predetermined weights - If set as None, it creates the new one
    "bias": None,  # Predetermined bias - If set as None, it creates the new one
    "correlation": 1,  # Correlation coefficient between features
    "noise_std": 0.01,  # Noise scale to check durability of the model
    "random_state": 42,  # Random seed
    "test_ratio": 0.3,  # Portion of the test set from total sample.
}

"""
 Training configuration 
"""

# User picked data type to use : [options] "file", "synthetic", "multi_collinear"
data_selection = "multi_collinear"  # Default is "synthetic"

# User defined params for training model
test_params = {
    "learning_rate": 0.01,  # Learning rate for gradient regression
    "epochs": 10000,  # Number of trainings
    "alpha": 0.001,  # Strength controller of regularization
    "rho": 0.015,  # L1 ratio
    "optimization": True  # Whether to find optimal cost option
}
