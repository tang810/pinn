import deepxde as dde


def create_network():
    """Defines the neural network architecture."""
    return dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
