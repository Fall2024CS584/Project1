import numpy as np

def generate_negative_data(range_x, noise_scale, size, num_features, seed):
    """
    Generate synthetic data with specified patterns:
    - The first half of the features have a monotonic trend.
    - The second half of the features have a linear decreasing (negative slope) pattern.

    Parameters:
        range_x - Range of feature values (min, max)
        noise_scale - Standard deviation of the noise
        size - Number of samples
        num_features - Number of features
        seed - Random seed for reproducibility

    Returns:
        X - Generated multi-dimensional feature dataset
        y - Target values
    """
    rng = np.random.default_rng(seed=seed)
    half_features = num_features // 2  # Calculate half the number of features
    
    # Generate X1 data with a clear linear trend
    X1 = np.zeros((size, half_features))
    for i in range(half_features):
        # Generate linear data from low to high to ensure a clear positive slope
        X1[:, i] = np.linspace(range_x[1], range_x[0], size) + rng.normal(loc=0, scale=noise_scale, size=size)###

    # Define X2 matrix as (size, num_features - half_features)
    X2 = np.zeros((size, num_features - half_features))

    # Generate data with a negative slope
    for i in range(X2.shape[1]):
        # Generate data from high to low with a negative slope
        X2[:, i] = np.linspace(range_x[1], range_x[0], size)
        # Add appropriate negative weights to ensure negative correlation between features and target values
        X2[:, i] += rng.normal(loc=0, scale=noise_scale, size=size)#####

    # Flip to ensure X2 is arranged from large to small
    X2 = np.flip(X2, axis=0)

    # Combine X1 and X2
    X = np.hstack((X1, X2))

    # Generate target values y, enhancing the negative correlation of negative slope features
    y = (2 * X1[:, 0] +    # Contribution of the first dimension multiplied by 2
         3 * X1[:, 1] +    # Contribution of the second dimension multiplied by 3
         4 * X1[:, 2] -    # Contribution of the third dimension multiplied by 4
         4 * np.sum(X2, axis=1) +  # Strong negative slope contribution from decreasing data
         rng.normal(loc=0, scale=noise_scale, size=size))  # Add noise
    
    return X, y