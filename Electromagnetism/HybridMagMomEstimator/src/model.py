import torch
import torch.nn as nn


class MagneticMomentNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, output_size)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        # Keep same behavior as original implementation: aggregate to one moment set.
        return torch.mean(x, dim=0, keepdim=True)
