import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    """Physics-Informed Neural Network model following power system physical constraints"""
    def __init__(self, num_nodes=10, hidden_dim=64, num_layers=4):
        super(PINN, self).__init__()
        self.num_nodes = num_nodes
        self.input_layer = nn.Linear(2, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.Tanh())
        self.output_layer = nn.Linear(hidden_dim, 3)
        self.M = torch.tensor(6.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  
        self.X_line = torch.tensor(0.1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  
        self.node_spacing = 100.0  
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, x):
        inputs = torch.cat([t.unsqueeze(1), x.unsqueeze(1)], dim=1)
        out = self.input_layer(inputs)
        out = torch.tanh(out)
        for layer in self.hidden_layers:
            out = layer(out)
        outputs = self.output_layer(out)
        delta_f = outputs[:, 0:1]
        delta_P_e = outputs[:, 1:2]
        delta_P_m = outputs[:, 2:3]
        return delta_f, delta_P_e, delta_P_m

    def compute_derivatives(self, t, x):
        t.requires_grad_(True)
        x.requires_grad_(True)
        delta_f, delta_P_e, delta_P_m = self.forward(t, x)
        df_dt = torch.autograd.grad(
            delta_f, t,
            grad_outputs=torch.ones_like(delta_f),
            create_graph=True,
            retain_graph=True
        )[0]
        delta_theta = torch.autograd.grad(
            delta_f.sum(), t,
            create_graph=True,
            retain_graph=True
        )[0].unsqueeze(1)
        return delta_f, delta_P_e, delta_P_m, df_dt, delta_theta