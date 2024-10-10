import numpy as np
import csv
import argparse

def generate_data(N, m, b, scale, rnge, seed, output_file):
    def linear_data_generator(m, b, rnge, N, scale, seed):
        rng = np.random.default_rng(seed=seed)
        sample = rng.uniform(low=rnge[0], high=rnge[1], size=(N, len(m)))
        ys = np.dot(sample, np.array(m).reshape(-1, 1)) + b
        noise = rng.normal(loc=0., scale=scale, size=ys.shape)
        return (sample, ys + noise)

    def write_data(filename, X, y):
        with open(filename, "w") as file:
            xs = [f"x_{n}" for n in range(X.shape[1])]
            header = xs + ["y"]
            writer = csv.writer(file)
            writer.writerow(header)
            for row in np.hstack((X, y)):
                writer.writerow(row)

    m = np.array(m)
    X, y = linear_data_generator(m, b, rnge, N, scale, seed)
    write_data(output_file, X, y)

# Calling the function with example parameters
generate_data(100, [3, 2], 5, 1.0, [-10, 10], 42, 'data.csv')
