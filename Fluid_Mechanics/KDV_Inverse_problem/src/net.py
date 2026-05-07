import os
os.environ["DDE_BACKEND"] = "pytorch"   # 小写且要放最前
import deepxde as dde

def create_network():
    net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot uniform")
    return net
