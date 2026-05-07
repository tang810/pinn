import torch
import torch.nn as nn
class MLP(nn.Module):
    """
    Multi-layer perceptron with tanh activation
    """
    def __init__(self, layers):
        super(MLP, self).__init__()
        
        modules = []
        for i in range(len(layers)-2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        
        self.net = nn.Sequential(*modules)
        
        # Initialize weights using Xavier/Glorot initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)
