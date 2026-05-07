import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=64, num_layers=4):
        super(PINN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
