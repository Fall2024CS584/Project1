import numpy as np

def generate_rotated_positive_data(range_x, noise_scale, size, num_features, seed, rotation_angle=45, mode=0):
    """
    Generate synthetic data with specified patterns:
    - The first half of the features have a monotonic trend.
    - The second half of the features have a wavy (slanted S-shaped) pattern adjusted by rotation.

    Parameters:
        range_x - Range of feature values (min, max)
        noise_scale - Standard deviation of the noise
        size - Number of samples
        num_features - Number of features
        seed - Random seed for reproducibility
        rotation_angle - Rotation angle to adjust the direction of the S-shape
        mode - Determines scaling factors to use

    Returns:
        X - Generated multi-dimensional feature dataset
        y - Target values
    """

    def scale_random_rows(X, size, scale_factors, seed=None):
        """
        Scale random rows of a given matrix X by specified scaling factors.

        Parameters:
            X - Input matrix
            size - Number of rows in the matrix, i.e., number of samples
            scale_factors - A list of scaling factors, e.g., [0.5, 0.7, 0.3], indicating factors for scaling some rows
            seed - Random seed for reproducibility

        Returns:
            Modified matrix X
        """
        rng = np.random.default_rng(seed=seed)
        remaining_indices = np.arange(size)  # Initialize with all row indices

        for scale in scale_factors:
            # Randomly select 1/n of rows for scaling
            selected_indices = rng.choice(remaining_indices, size // len(scale_factors), replace=False)
            X[selected_indices, :] *= scale  # Scale by the scaling factor
            # Update the remaining indices
            remaining_indices = np.setdiff1d(remaining_indices, selected_indices)

        return X

    rng = np.random.default_rng(seed=seed)
    half_features = num_features // 2  # Calculate half the number of features
    
    # Generate X1 data
    X1 = rng.uniform(low=range_x[0], high=range_x[1], size=(size, half_features))
    if mode == 0:
        scale_factors = [0.5, 0.7, 0.3]  # Scaling factors needed
    else:
        scale_factors = [0.8, 0.4]
    X1 = scale_random_rows(X1, X1.shape[0], scale_factors, seed=42)

    # Generate S-shaped data for the second half
    X2 = rng.uniform(low=range_x[0], high=range_x[1], size=(size, num_features - half_features))
    for i in range(X2.shape[1]):
        # Create standard S-shaped curve
        X2[:, i] = np.sin(X2[:, i] / 2) * 10 + 0.5 * X2[:, i]
    X2 = scale_random_rows(X2, X2.shape[0], scale_factors, seed=52)

    # Create rotation matrix, convert angle to radians
    theta = np.radians(rotation_angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    # Rotate the first two dimensions of S-shaped data to make it slanted
    rotated_X2 = X2[:, :2] @ rotation_matrix  # Rotate only the first two columns since the matrix is 2x2
    X2[:, :2] = rotated_X2  # Assign the rotated part back

    # Combine the two parts
    X = np.hstack((X1, X2))
    
    # Generate target values y, adjusted by the contribution of each dimension
    y = (2 * X1[:, 0] +    # Contribution of the first dimension multiplied by 2
        3 * X1[:, 1] +    # Contribution of the second dimension multiplied by 3
        4 * X1[:, 2] +    # Contribution of the third dimension multiplied by 4
        np.sum(np.sin(X2), axis=1) +  # Contribution of S-shaped data
        rng.normal(loc=0, scale=noise_scale, size=size))  # Add noise
        
    return X, y
