
import torch
import torch.nn as nn
import torch.nn.init as init

class STPinn(nn.Module):
    """
    Spatio-Temporal PINN for Rainfall Prediction.
    Input: (x, y, t)
    Output: Rainfall Intensity R
    """
    def __init__(self, layers=[3, 50, 50, 50, 50, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self._init_weights()

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x

def build_model(device):
    """Constructs the PINN model."""
    model = STPinn().to(device)
    return model
