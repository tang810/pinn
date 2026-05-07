import torch
from collections import OrderedDict


# neural network
class FCNet(torch.nn.Module):
    def __init__(self, num_ins=3,
                 num_outs=3,
                 num_layers=10,
                 hidden_size=50,
                 activation=torch.nn.Tanh):
        """
        初始化全连接神经网络模型。

        参数:
        - num_ins: 输入特征的数量，默认为3。
        - num_outs: 输出特征的数量，默认为3。
        - num_layers: 隐藏层的数量，默认为10。
        - hidden_size: 隐藏层的神经元数量，默认为50。
        - activation: 激活函数，默认为双曲正切函数。
        """
        super(FCNet, self).__init__()

        layers = [num_ins] + [hidden_size] * num_layers + [num_outs]
        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = activation

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x: 输入张量。

        返回:
        - out: 神经网络的输出。
        """
        out = self.layers(x)
        return out
