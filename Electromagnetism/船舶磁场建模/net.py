# Import libraries
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from collections import OrderedDict
# from pinnsformer import PINNsformer
# Import the summary writer 
from torch.utils.tensorboard import SummaryWriter# Create an instance of the object 
writer = SummaryWriter(log_dir='./forward_tensorboard_event_file')

import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from datetime import datetime
import scipy.io
import os
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # Run on CPU
# device = ('cuda:0')

# Seeds
torch.manual_seed(123)
np.random.seed(123)


def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,grad_outputs=torch.ones_like(outputs), create_graph=True)

class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)
    
# Define a deep neural network class
class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # Crear a list to hold the layers of the network
        layer_list = []
        for i in range(len(layers)-2):
        # add linear layer
            layer_list.append(('layer_%d' % i, nn.Linear(layers[i], layers[i+1])))
            # add activation function WaveAct
            layer_list.append(('activation_%d' % i, nn.ReLU()))
        # layer_list.append(('layer_%d' % i, nn.Linear(layers[-3], layers[-2])))
        # layer_list.append(('activation_%d' % i, nn.Sigmoid()))
        # add final linear layer
        layer_list.append(('layer_%d' % (len(layers)-2), nn.Linear(layers[-2], layers[-1])))
        
        layerDict = OrderedDict(layer_list)

        # using nn.Sequrntial to deploy layers
        self.layers = nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out
    def _initialize_weights(self):
        # Iterate through layers in self.layers
        for name, layer in self.layers.named_children():
            if isinstance(layer, nn.Linear):
                # Xavier/Glorot initialization for weight
                nn.init.xavier_uniform_(layer.weight)
                # Initialize bias to zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)




