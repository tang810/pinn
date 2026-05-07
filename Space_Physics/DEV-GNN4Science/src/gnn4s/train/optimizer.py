import torch

class AdamOptimizer(torch.optim.Adam):
    def __init__(self, parameters, lr):
        super(AdamOptimizer, self).__init__(parameters, lr=lr)

class StepLR(torch.optim.lr_scheduler.StepLR):
    def __init__(self, optimizer, step_size, gamma):
        super(StepLR, self).__init__(optimizer, step_size=step_size, gamma=gamma)

class ExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma):
        super(ExponentialLR, self).__init__(optimizer, gamma=gamma)