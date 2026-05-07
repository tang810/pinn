import os
import numpy as np

def create_directories(args):
    os.makedirs(args.output_dir, exist_ok=True)

def LHSample(D, bounds, N):
    """
    Latin Hypercube Sampling.
    D: dimension
    bounds: list of [min,max] for each dim
    N: number of samples
    Returns: (N, D) array
    """
    result = np.empty((N, D))
    temp = np.empty((N, D))
    d = 1.0 / N
    for i in range(D):
        for j in range(N):
            temp[j, i] = np.random.uniform(low=j * d, high=(j + 1) * d)
        np.random.shuffle(temp[:, i])

    for i in range(D):
        result[:, i] = temp[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    return result