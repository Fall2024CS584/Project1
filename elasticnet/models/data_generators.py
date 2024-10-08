import numpy as np
from scipy.signal import sawtooth

def linear_data_generator1(m, b, range_, N, seed):
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(low=range_[0], high=range_[1], size=N)
    ys = m * sample + b
    noise = rng.normal(loc=0., scale=3, size=N)
    return sample, ys + noise

def linear_data_generator2(m, b, range_, N, seed):
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(low=range_[0], high=range_[1], size=(N, m.shape[0]))
    ys = np.dot(sample, np.reshape(m, (-1, 1))) + b
    noise = rng.normal(loc=0., scale=50, size=ys.shape)
    return sample, ys.flatten()

def nonlinear_data_generator1(m, b, range_, N, seed):
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(low=range_[0], high=range_[1], size=N)
    ys = np.exp(m * sample) + b
    noise = rng.normal(loc=0, scale=0.5, size=N)
    return sample, ys + noise

def generate_collinear_data(range_, noise_scale, size, seed):
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(low=range_[0], high=range_[1], size=size)
    new_col = rng.normal(loc=0, scale=0.01, size=sample.shape[0])
    new_col = np.reshape(new_col, (-1, 1))
    new_sample = np.hstack((sample, new_col))
    m = rng.integers(low=-10, high=10, size=(new_sample.shape[1], 1))
    ys = np.dot(new_sample, m)
    noise = rng.normal(loc=0, scale=noise_scale, size=ys.shape)
    return new_sample, ys + noise

def generate_periodic_data(period, amplitude, range_, noise_scale, size, seed):
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(low=range_[0], high=range_[1], size=size)
    ys = amplitude * sawtooth(sample * 2 * np.pi / period - 1.47)
    noise = rng.normal(loc=0, scale=noise_scale, size=ys.shape)
    return sample, ys + noise

def generate_higher_dim_data(range_, noise_scale, size, seed):
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(low=range_[0], high=range_[1], size=size)
    ys = np.power(sample[:, 0], 2) + np.power(sample[:, 1], 3) - np.linalg.norm(sample, axis=1)
    noise = rng.normal(loc=0, scale=noise_scale, size=ys.shape)
    return sample, ys + noise
