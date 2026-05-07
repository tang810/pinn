import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax, dense_to_sparse
from torch_scatter import scatter
from typing import Tuple, Optional
import functools
import math

from .mlp import MLP, Activation

class PosTransLayer(MessagePassing):
    """Involving the edge feature and updating position feature. Multiply Msg."""

    _alpha: OptTensor

    def __init__(self, deg_channels_num: int, pos_channels_num: int, out_channels_num: int,
                 heads_num: int=1, edge_dim: Optional[int] = None, bias: bool=True, 
                 dropout_rate: float=0., activation=None, attn_clamp: bool=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(PosTransLayer, self).__init__(node_dim=0, **kwargs)

        self.deg_channels_num = deg_channels_num
        self.pos_channels_num = pos_channels_num
        self.out_channels_num = out_channels_num
        self.heads_num = heads_num
        self.edge_dim = edge_dim
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.actv = activation
        self.attn_clamp = attn_clamp

        self.in_channels_num = deg_channels_num + pos_channels_num
        
        self.lin_key = nn.Linear(self.in_channels_num, self.heads_num*self.out_channels_num)
        self.lin_query = nn.Linear(self.in_channels_num, self.heads_num*self.out_channels_num)
        self.lin_value = nn.Linear(self.in_channels_num, self.heads_num*self.out_channels_num)

        self.lin_edge0 = nn.Linear(self.edge_dim, self.heads_num*self.out_channels_num, bias=False)
        self.lin_edge1 = nn.Linear(self.edge_dim, self.heads_num*self.out_channels_num, bias=False)

        self.lin_pos = nn.Linear(self.heads_num*self.out_channels_num, self.pos_channels_num, bias=False)

        self.lin_skip = nn.Linear(self.deg_channels_num, self.heads_num*self.out_channels_num, bias=self.bias)
        self.norm1 = nn.GroupNorm(num_groups=min(self.heads_num*self.out_channels_num // 4, 32),
                                  num_channels=self.heads_num*self.out_channels_num, eps=1e-6)
        self.norm2 = nn.GroupNorm(num_groups=min(self.heads_num*self.out_channels_num // 4, 32),
                                  num_channels=self.heads_num*self.out_channels_num, eps=1e-6)
        
        self.mlp = MLP(input_dim=self.heads_num*self.out_channels_num,
                       output_dim=self.heads_num*self.out_channels_num,
                       hidden_dim=self.heads_num*self.out_channels_num,
                       layer_num=1, activation=self.actv)

        if self.actv is None:
            self.actv = nn.LeakyReLU(negative_slope=0.2)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()
        self.lin_pos.reset_parameters()

    def forward(self, deg: OptTensor,
                pos: torch.Tensor,
                edge_index: Adj,
                edge_attr: OptTensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward propagation
        """

        x_f = torch.cat([deg, pos], -1)
        query = self.lin_query(x_f).view(-1, self.heads_num, self.out_channels_num)
        key = self.lin_key(x_f).view(-1, self.heads_num, self.out_channels_num)
        value = self.lin_value(x_f).view(-1, self.heads_num, self.out_channels_num)
        
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x, out_pos = self.propagate(edge_index, query=query, key=key, value=value, pos=pos,
                                        edge_attr=edge_attr, size=None)

        out_x = out_x.view(-1,self.heads_num*self.out_channels_num)

        # skip connection for x
        x_r = self.lin_skip(deg)
        out_x = (out_x + x_r) / math.sqrt(2)
        out_x = self.norm1(out_x)

        # FFN
        out_x = (out_x + self.mlp(out_x)) / math.sqrt(2)
        out_x = self.norm2(out_x)

        # skip connection for pos
        out_pos = pos + torch.tanh(pos + out_pos)

        return out_x, out_pos

    def message(self, query_i: torch.Tensor, key_j: torch.Tensor, value_j: torch.Tensor,
                pos_j: torch.Tensor,
                edge_attr: OptTensor,
                index: torch.Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:

        edge_attn = self.lin_edge0(edge_attr).view(-1, self.heads_num, self.out_channels_num)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.out_channels_num)
        if self.attn_clamp:
            alpha = alpha.clamp(min=-5., max=5.)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * self.lin_edge1(edge_attr).view(-1, self.heads_num, self.out_channels_num)
        msg = msg * alpha.view(-1, self.heads_num, 1)

        # node position message
        pos_msg = pos_j * self.lin_pos(msg.reshape(-1, self.heads_num * self.out_channels_num))

        return msg, pos_msg

    def aggregate(self, inputs: Tuple[torch.Tensor, torch.Tensor], index: torch.Tensor,
                  ptr: Optional[torch.Tensor]=None,
                  dim_size: Optional[int]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if ptr is not None:
            raise NotImplementedError("Not implement Ptr in aggregate")
        else:
            return (scatter(inputs[0], index, 0, dim_size=dim_size, reduce=self.aggr),
                    scatter(inputs[1], index, 0, dim_size=dim_size, reduce="mean"))

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels_num, self.out_channels_num, self.heads_num)

class PosGNN(nn.Module):
    def __init__(self, time_embd_dim: int=None, deg_channels_num: int=64, pos_channels_num: int=64, out_channels_num: int=64,
                 max_node_num: int=64, edge_dim: int=64, graph_layer_type: str=None, graph_layer_num: int=3, heads_num=4,
                 activation=None, dropout_rate: float=0.1, attn_clamp: bool=False):
        super().__init__()

        self.time_embd_dim = time_embd_dim
        self.deg_channels_num = deg_channels_num
        self.pos_channels_num = pos_channels_num
        self.out_channels_num = out_channels_num
        self.edge_dim = edge_dim
        self.max_node_num = max_node_num
        self.actv = activation
        self.graph_layer_type = graph_layer_type
        self.graph_layer_num = graph_layer_num
        self.heads_num = heads_num
        self.dropout_rate = dropout_rate
        self.attn_clamp = attn_clamp

        if self.time_embd_dim is not None:
            self.dense_node0 = nn.Linear(self.time_embd_dim, self.deg_channels_num)
            self.dense_node1 = nn.Linear(self.time_embd_dim, self.pos_channels_num)
            self.dense_edge0 = nn.Linear(self.time_embd_dim, self.edge_dim)
            self.dense_edge1 = nn.Linear(self.time_embd_dim, self.edge_dim)
        
        self.node_convs = nn.ModuleList()
        self.edge_convs = nn.ModuleList()
        if self.graph_layer_type is None:
            self.graph_layer_type = 'PosTransLayer'
        for i in range(graph_layer_num):
            if i == 0:
                self.node_convs.append(
                    eval(self.graph_layer_type)(
                        self.deg_channels_num, self.pos_channels_num, self.out_channels_num//self.heads_num,
                        self.heads_num, edge_dim=self.edge_dim*2,
                        activation=self.actv, attn_clamp=self.attn_clamp))
            else:
                self.node_convs.append(
                    eval(self.graph_layer_type)(
                        self.out_channels_num, self.pos_channels_num, self.out_channels_num//self.heads_num,
                        self.heads_num, edge_dim=self.edge_dim*2,
                        activation=self.actv, attn_clamp=self.attn_clamp))
            self.edge_convs.append(nn.Linear(self.out_channels_num, self.edge_dim*2))

        self.dropout = nn.Dropout(self.dropout_rate)

        self.edge_final_layer = nn.Linear(self.edge_dim*2+self.out_channels_num, self.edge_dim)
        
    def forward(self, x_deg, x_pos, edge_idx, dense_ori, dense_spd, dense_idx, time_embd=None):
        """ Forward propagation
        
        Args:
            x_deg: node degree feature [batch_num*node_num, deg_channels_num]
            x_pos: node rwpe feature [batch_num*node_num, pos_channels_num]
            edge_idx: [2, edge_length]
            dense_ori: edge feature [batch_num, node_num, node_num, features_num//2]
            dense_spd: edge shortest path distance feature [batch_num, node_num, node_num, features_num//2]
            dense_idx: ?  [batch_num*max_node_num]
            time_embd: [batch_num, time_embd_dim]
        """
        batch_num, node_num, _, _ = dense_ori.shape

        if time_embd is not None:
            # dense_ori: [batch_num, , , deg_channels_num]
            # dense_spd: [batch_num, , , pos_channels_num]
            dense_ori = dense_ori + self.dense_edge0(self.actv(time_embd))[:, None, None, :]
            dense_spd = dense_spd + self.dense_edge1(self.actv(time_embd))[:, None, None, :]

            # time_embd: [batch_num*max_node_num, time_embd_dim]
            # x_deg: [batch_num*max_node_num, edge_dim]
            # x_pos: [batch_num*max_node_num, edge_dim]
            time_embd = time_embd.unsqueeze(1).repeat(1, self.max_node_num, 1)
            time_embd = time_embd.reshape(-1, time_embd.shape[-1])
            x_deg = x_deg + self.dense_node0(self.actv(time_embd))
            x_pos = x_pos + self.dense_node1(self.actv(time_embd))
        
        # dense_edge: [batch_num, , , deg_channels_num+pos_channels_num]
        dense_edge = torch.cat([dense_ori, dense_spd], dim=-1)
        edge_attr_ori = dense_edge
        h_deg = x_deg
        h_pos = x_pos

        for i in range(self.graph_layer_num):
            # h_edge: ? [batch_num*max_node_num, deg_channels_num+pos_channels_num]
            h_edge = dense_edge[dense_idx]

            # update node feature
            h_deg, h_pos = self.node_convs[i](h_deg, h_pos, edge_idx, h_edge)
            h_deg = self.dropout(h_deg)
            h_pos = self.dropout(h_pos)
            
            # update dense edge feature
            h_dense_node = h_deg.reshape(batch_num, node_num, -1)
            # edge_attr_cur: [batch_num, node_num, node_num, features_num]
            edge_attr_cur = h_dense_node.unsqueeze(1) + h_dense_node.unsqueeze(2)
            dense_edge = (dense_edge + self.actv(self.edge_convs[i](edge_attr_cur))) / math.sqrt(2.)
            dense_edge = self.dropout(dense_edge)
        
        # Concat edge attribute
        h_dense_edge = torch.cat([edge_attr_ori, dense_edge], dim=-1)
        h_dense_edge = self.edge_final_layer(h_dense_edge).permute(0, 3, 1, 2)

        return h_dense_edge

class PGSN(nn.Module):
    """Position enhanced graph score network."""

    def __init__(self, embd_type: str='positional', features_num: int=256, actv_type: str='silu',
                 node_num_cond: bool=False, max_node_num: int=125, 
                 channels_num: int=1, centered: bool=True, rw_depth: int=32, edge_th: float=0.2, 
                 graph_layer_type: str='PosTransLayer', graph_layer_num: int=1,
                 heads_num: int=8, dropout_rate: float=0.1, attn_clamp: bool=False,
                 device: str='cpu'):
        super().__init__()

        self.embd_type = embd_type
        self.features_num = features_num
        self.actv_type = actv_type
        self.node_num_cond = node_num_cond
        self.max_node_num = max_node_num
        self.channels_num = channels_num
        self.centered = centered
        self.rw_depth = rw_depth
        self.edge_th = edge_th
        self.graph_layer_type = graph_layer_type
        self.graph_layer_num = graph_layer_num
        self.heads_num = heads_num
        self.dropout_rate = dropout_rate
        self.attn_clamp = attn_clamp        
        self.device = device
        
        # timestep/noise_level embedding; only for continuous training
        if embd_type == 'positional':
            self.embd_dim = self.features_num
        else:
            raise ValueError(f'embedding type {embd_type} unknown.')

        # timestep embedding layers
        self.actv = Activation(self.actv_type)
        self.time_embd_layer = MLP(
            input_dim=self.embd_dim, output_dim=self.features_num*4,
            hidden_dim=self.features_num*4, layer_num=1, 
            activation=self.actv, device=self.device)
        
        # graph size condition embedding
        if node_num_cond:
            self.size_onehot = functools.partial(nn.functional.one_hot, num_classes=self.max_node_num+1)
            self.node_num_embd_layer = MLP(
                input_dim=self.max_node_num+1, output_dim=self.features_num*4,
                hidden_dim=self.features_num*4, layer_num=1, 
                activation=self.actv, device=self.device)
        
        assert self.channels_num == 1, "Without edge features."

        # degree onehot
        self.degree_max = self.max_node_num // 2
        self.degree_onehot = functools.partial(
            nn.functional.one_hot,
            num_classes=self.degree_max + 1)

        # project edge features
        self.edge_ori_dense_layer = nn.Conv2d(
            self.channels_num, self.features_num//2,
            kernel_size=1, stride=1, dilation=1, padding=0, bias=True)
        self.edge_spd_dense_layer = nn.Conv2d(
            self.rw_depth + 1, self.features_num//2,
            kernel_size=1, stride=1, dilation=1, padding=0, bias=True)

        # project node features
        self.deg_channels_num = self.features_num
        self.pos_channels_num = self.features_num // 2
        self.node_deg_dense_layer = nn.Linear(self.degree_max + 1, self.deg_channels_num)
        self.node_pos_dense_layer = nn.Linear(self.rw_depth, self.pos_channels_num)

        # GNN
        self.graph_layer = PosGNN(
            time_embd_dim=self.features_num*4, deg_channels_num=self.deg_channels_num,
            pos_channels_num=self.pos_channels_num, out_channels_num=self.features_num,
            max_node_num=self.max_node_num, edge_dim=self.features_num//2,
            graph_layer_type=self.graph_layer_type, graph_layer_num=graph_layer_num, heads_num=self.heads_num,
            activation=self.actv, dropout_rate=dropout_rate, attn_clamp=self.attn_clamp)

        # output
        self.final_layer0 = nn.Conv2d(
            self.features_num//2, self.features_num//2,
            kernel_size=1, stride=1, dilation=1, padding=0, bias=True)
        self.final_layer1 = nn.Conv2d(
            self.features_num//2, self.channels_num,
            kernel_size=1, stride=1, dilation=1, padding=0, bias=True)

        self.to(self.device)

    @torch.no_grad()
    def timestep_embedding(self, timesteps, embd_dim, max_positions=10000):
        # Sinusoidal positional embeddings
        # magic number 10000 is from transformers
        time_embd = math.log(max_positions) / (embd_dim//2 - 1)
        time_embd = torch.exp(torch.arange(embd_dim//2, dtype=torch.float32, device=timesteps.device) * -time_embd)
        time_embd = timesteps.float()[:, None] * time_embd[None, :]
        time_embd = torch.cat([torch.sin(time_embd), torch.cos(time_embd)], dim=1)
        if embd_dim % 2 == 1: # zero pad
            time_embd = F.pad(time_embd, (0, 1), mode='constant')
        return time_embd
    
    @torch.no_grad()
    def get_rw_feat(self, k_step, dense_adj):
        """Compute k_step Random Walk for given dense adjacency matrix."""

        rw_list = []
        deg = dense_adj.sum(-1, keepdims=True)
        AD = dense_adj / (deg + 1e-8)
        rw_list.append(AD)

        for _ in range(k_step):
            rw = torch.bmm(rw_list[-1], AD)
            rw_list.append(rw)
        rw_map = torch.stack(rw_list[1:], dim=1)  # [B, k_step, N, N]

        rw_landing = torch.diagonal(rw_map, offset=0, dim1=2, dim2=3)  # [B, k_step, N]
        rw_landing = rw_landing.permute(0, 2, 1)  # [B, N, rw_depth]

        # get the shortest path distance indices
        tmp_rw = rw_map.sort(dim=1)[0]
        spd_ind = (tmp_rw <= 0).sum(dim=1)  # [B, N, N]

        spd_onehot = torch.nn.functional.one_hot(spd_ind, num_classes=k_step+1).to(torch.float)
        spd_onehot = spd_onehot.permute(0, 3, 1, 2)  # [B, kstep, N, N]

        return rw_landing, spd_onehot

    def forward(self, aa, x, mask, time_cond):
        
        # time embedding
        timesteps = time_cond
        time_embd = self.timestep_embedding(timesteps, self.features_num)
        time_embd = self.time_embd_layer(time_embd)
        
        if self.node_num_cond:
            with torch.no_grad():
                node_mask = mask.squeeze(1)[:,0,:].clone()
                node_mask[:,0] = 1  # [B, N]
                node_num = torch.sum(node_mask, dim=-1)  # [B]
                node_num = self.size_onehot(node_num.to(torch.long)).to(torch.float)
            node_num_embd = self.node_num_embd_layer(node_num)
            time_embd = time_embd + node_num_embd

        if not self.centered:
            # rescale the input data to [-1, 1]
            x = x * 2. - 1.

        with torch.no_grad():
            # continuous-valued graph adjacency matrices
            cont_adj = ((x + 1.) / 2.).clone()
            cont_adj = (cont_adj * mask).squeeze(1)  # [B, N, N]
            cont_adj = cont_adj.clamp(min=0., max=1.)
            if self.edge_th > 0.:
                cont_adj[cont_adj < self.edge_th] = 0.

            # discretized graph adjacency matrices
            adj = x.squeeze(1).clone()  # [B, N, N]
            adj[adj >= 0.] = 1.
            adj[adj < 0.] = 0.
            adj = adj * mask.squeeze(1)

        # extract RWSE and Shortest-Path Distance
        x_pos, spd_onehot = self.get_rw_feat(self.rw_depth, adj)
        
        # edge [B, N, N, F]
        dense_edge_ori = self.edge_ori_dense_layer(x).permute(0, 2, 3, 1)
        dense_edge_spd = self.edge_spd_dense_layer(spd_onehot).permute(0, 2, 3, 1)

        # Use Degree as node feature
        x_deg = torch.sum(cont_adj, dim=-1)  # [B, N]
        x_deg = x_deg.clamp(max=float(self.degree_max))
        x_deg = self.degree_onehot(x_deg.to(torch.long)).to(torch.float)  # [B, N, max_node]
        x_deg = self.node_deg_dense_layer(x_deg)  # projection layer [B, N, nf]

        # pos encoding
        x_pos = self.node_pos_dense_layer(x_pos)

        # Dense to sparse node [BxN, -1]
        x_deg = x_deg.reshape(-1, self.deg_channels_num)
        x_pos = x_pos.reshape(-1, self.pos_channels_num)
        dense_index = cont_adj.nonzero(as_tuple=True)
        edge_index, _ = dense_to_sparse(cont_adj)

        # Run GNN layers
        h_dense_edge = self.graph_layer(
            x_deg, x_pos, edge_index, dense_edge_ori, dense_edge_spd, dense_index, time_embd)

        # Output
        h = self.actv(self.final_layer0(self.actv(h_dense_edge)))
        h = self.final_layer1(h)

        # make edge estimation symmetric
        h = (h + h.transpose(2, 3)) / 2. * mask

        return h
