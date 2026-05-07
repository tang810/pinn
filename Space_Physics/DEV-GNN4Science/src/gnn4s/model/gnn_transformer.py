import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_scatter
from torch_geometric.data import Data as PyGData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from types import SimpleNamespace

from .mlp import MLP
from .normalizer import Normalizer

class GCNConvAttention(MessagePassing):
    def __init__(self, input_dim, output_dim, activation=None):
        super(GCNConvAttention, self).__init__(aggr='add')  # Use "add" as the aggregation function
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actv = activation

        if self.actv is None:
            self.actv = nn.LeakyReLU(negative_slope=0.2)
        
        # Linear transformation layers for query, key, and value
        self.lin_que = nn.Linear(self.input_dim, self.output_dim)
        self.lin_key = nn.Linear(self.input_dim, self.output_dim)
        self.lin_val = nn.Linear(self.input_dim, self.output_dim)
        self.lin_edge = nn.Linear(self.input_dim, self.output_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_que.reset_parameters()
        self.lin_val.reset_parameters()
        self.lin_edge.reset_parameters()
        
    def forward(self, graph):
        # Compute normalized node degrees
        edge_index = graph.edge_index
        deg = degree(edge_index[0], graph.node_feat.size(0), dtype=graph.node_feat.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]

        # Linear transformation of features
        node_que = self.lin_que(graph.node_feat)
        node_key = self.lin_key(graph.node_feat)
        node_val = self.lin_val(graph.node_feat)

        # Message passing
        edge_feat = graph.edge_feat
        node_feat = self.propagate(edge_index, que=node_que, key=node_key, val=node_val,
                                   edge_feat=edge_feat, edge_idx=edge_index)
        return node_feat

    def message(self, que_i: torch.Tensor, key_j: torch.Tensor, val_j: torch.Tensor,
                edge_feat: torch.Tensor, edge_idx: torch.Tensor):
        # que_i: Query features of source nodes
        # key_j: Key features of target nodes
        # val_j: Value features of target nodes
        # edge_feat: Edge attributes
        edge_attn = self.lin_edge(edge_feat)

        # attention weights
        alpha = que_i * (key_j + edge_attn)
        alpha = alpha.sum(-1,keepdims=True) / (alpha.shape[-1]**0.5)

        # softmax over edges for each node
        src, dst = edge_idx[0], edge_idx[1]
        alpha = torch_scatter.scatter_softmax(alpha, src, dim=0)

        # weighted values
        msg = alpha * (val_j + edge_attn)
        return msg

    def update(self, aggr_out):
        # aggr_out is the sum of messages passed from all neighboring nodes
        return aggr_out

class Encoder(nn.Module):
    def __init__(self, node_dim=3, edge_dim=4, hidden_dim=128):
        super(Encoder, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.node_embd = MLP(self.node_dim, self.hidden_dim, self.hidden_dim,
                             layer_num=2, norm=True, activation=nn.ReLU())
        self.edge_embd = MLP(self.edge_dim, self.hidden_dim, self.hidden_dim,
                             layer_num=2, norm=True, activation=nn.ReLU())
        
    def forward(self, graph):
        node_feat = self.node_embd(graph.node_feat)
        edge_feat = self.edge_embd(graph.edge_feat)
        return PyGData(node_feat=node_feat, edge_feat=edge_feat, 
                       edge_index=graph.edge_index)

class Processer(nn.Module):
    def __init__(self, hidden_dim=128):
        super(Processer, self).__init__()
        self.hidden_dim = hidden_dim

        self.conv = GCNConvAttention(self.hidden_dim, self.hidden_dim)
        self.mlp = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim,
                       layer_num=2, norm=True, activation=nn.ReLU())

    def forward(self, graph):
        node_feat = graph.node_feat
        node_feat_update = self.conv(graph)
        node_feat_update = self.mlp(node_feat_update)
        node_feat = node_feat + node_feat_update
        return PyGData(node_feat=node_feat, edge_feat=graph.edge_feat, 
                       edge_index=graph.edge_index)

class Decoder(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=3):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.mlp = MLP(self.hidden_dim, self.output_dim, self.hidden_dim,
                       layer_num=2, activation=nn.ReLU())

    def forward(self, graph):
        return self.mlp(graph.node_feat)

class GCNTransformer(nn.Module):
    def __init__(self, var_dict, node_dim: int=3, edge_dim: int=4,
                 output_dim: int=3, hidden_dim=128, block_num: int=1) -> None:
        super(GCNTransformer, self).__init__()
        """ MeshGraphNet model
        Args:
            node_dim: dimension of node input features
            edge_dim: dimension of edge input features
            output_dim: dimension of output
            block_num: number of hidden block
        """
        self.var_dict = var_dict
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.block_num = block_num
        
        #self._node_normalizer = Normalizer(size=self.node_dim)
        #self._edge_normalizer = Normalizer(size=self.edge_dim)
        
        self.encoder = Encoder(node_dim=self.node_dim, edge_dim=self.edge_dim,
                               hidden_dim=self.hidden_dim)
        
        block = []
        for _ in range(self.block_num):
            block.append(Processer(hidden_dim=self.hidden_dim))
        self.block = nn.ModuleList(block)

        self.decoder = Decoder(hidden_dim=self.hidden_dim, output_dim=output_dim)
    
    def get_model_config(self):
        """ get model configiration """
        model_config_dict = {}
        model_config_dict['node_dim'] = self.node_dim
        model_config_dict['edge_dim'] = self.edge_dim
        model_config_dict['output_dim'] = self.output_dim
        model_config_dict['hidden_dim'] = self.hidden_dim
        model_config_dict['block_num'] = self.block_num
        model_config_namespace = SimpleNamespace(**model_config_dict)
        return model_config_namespace

    def forward(self, graph: PyGData):
        """ Forward propogation
        Args:
            graph: graph data
        """
        # pre-process
        node_feat = torch.cat([graph.node_pos,graph.node_attr], dim=-1)
        graph.node_feat = node_feat
        # graph.node_feat = self._node_normalizer(graph.node_feat, self.training)
        
        send_idx, recv_idx = graph.edge_index
        node_pos_rel = graph.node_pos[send_idx,:] - graph.node_pos[recv_idx,:]
        edge_feat = torch.cat([node_pos_rel, torch.norm(node_pos_rel,dim=1,keepdim=True)], -1)
        graph.edge_feat = edge_feat
        # graph.edge_feat = self._edge_normalizer(graph.edge_feat, self.training)
        
        # encode
        graph = self.encoder(graph)
        
        # process
        for i in range(len(self.block)):
            graph = self.block[i](graph)
        
        # decode
        pre = self.decoder(graph)
        
        # post-process
        node_pre = {}
        k = 0
        for key in self.var_dict['node_label']:
            node_pre[key] = pre[:,k:k+1]
            k += 1
        return node_pre