import torch
import torch.nn as nn
import math

from .mlp import MLP, Activation

class DenseConv(nn.Module):
    """ Dense Convolution layer
    See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels: int, out_channels: int, improved: bool=False,
                 bias: bool=True) -> None:
        super(DenseConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor=None,
                add_loop: bool=True) -> torch.Tensor:
        """
        Args:
            x: Node feature tensor, [batch_size, nodes_num_max, features_dim]
            adj: Adjacency tensor, [batch_size, nodes_num_max, nodes_num_max]
            mask: Mask matrix, [batch_size, nodes_num_max]
            add_loop: If the layer add self-loops to the adjacency matrices
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        batch_size, nodes_num_max, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(nodes_num_max, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out += self.bias

        if mask is not None:
            out *= mask.view(batch_size, nodes_num_max, 1).to(x.dtype)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class DenseAttention(torch.nn.Module):
    """ Graph Multi-Head Attention (GMH)
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, heads_num: int=4,
                 conv_type: str='GCN', actv=nn.Tanh()) -> None:
        super(DenseAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.heads_num = heads_num
        self.conv_type = conv_type
        self.actv = actv

        self.softmax_dim = 2
        if self.conv_type=='GCN':
            self.attn_q = DenseConv(self.input_dim, self.hidden_dim)
            self.attn_k = DenseConv(self.input_dim, self.hidden_dim)
            self.attn_v = DenseConv(self.input_dim, self.output_dim)
        elif self.conv_type=='MLP':
            layer_num = 1
            self.attn_q = MLP(input_dim=self.input_dim, output_dim=self.hidden_dim,
                             hidden_dim=2*self.hidden_dim, layer_num=layer_num, 
                             activation=self.actv)
            self.attn_k = MLP(input_dim=self.input_dim, output_dim=self.hidden_dim,
                             hidden_dim=2*self.hidden_dim, layer_num=layer_num, 
                             activation=self.actv)
            self.attn_v = DenseConv(self.input_dim, self.output_dim)
        else:
            raise NotImplementedError(f'{self.conv_type} not implemented.')
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, attn_mask: torch.Tensor=None) -> None:
        if self.conv_type=='GCN':
            q = self.attn_q(x, adj) 
            k = self.attn_k(x, adj) 
        else:
            q = self.attn_q(x) 
            k = self.attn_k(x)

        v = self.attn_v(x, adj)
        dim_split = self.hidden_dim // self.heads_num
        q_ = torch.cat(q.split(dim_split, 2), 0)
        k_ = torch.cat(k.split(dim_split, 2), 0)

        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask for _ in range(self.heads_num)], 0)
            attn_score = q_.bmm(k_.transpose(-1,-2))/math.sqrt(self.output_dim)
            a = self.actv(attn_mask + attn_score)
        else:
            a = self.actv(q_.bmm(k_.transpose(-1,-2))/math.sqrt(self.output_dim))

        # a: [batch_size*heads_num, nodes_num_max, nodes_num_max] ->
        #    [heads_num, batch_size, nodes_num_max, nodes_num_max] ->
        #    [batch_size, nodes_num_max, nodes_num_max]
        a = a.view(-1, *adj.shape)
        a = a.mean(dim=0)
        a = (a + a.transpose(-1,-2))/2 

        return v, a

class DenseAttentionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, lin_layer_num: int,
                 attn_input_dim: int, attn_output_dim: int, attn_hidden_dim: int, attn_heads_num: int=4,
                 conv_type: str='GCN', actv=nn.Tanh()) -> None:
        super(DenseAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lin_layer_num = lin_layer_num
        self.attn_input_dim = attn_input_dim
        self.attn_output_dim = attn_output_dim
        self.attn_hidden_dim = attn_hidden_dim
        self.attn_heads_num = attn_heads_num
        self.conv_type = conv_type
        self.actv = actv

        if isinstance(self.actv, str):
            self.actv = Activation(self.actv)

        self.attn = torch.nn.ModuleList()
        for i in range(self.input_dim):
            self.attn.append(
                DenseAttention(input_dim=self.attn_input_dim, output_dim=self.attn_output_dim,
                               hidden_dim=self.attn_hidden_dim, heads_num=attn_heads_num, conv_type=self.conv_type))

        self.lin_hidden_dim = 2*max(self.input_dim, self.output_dim)
        self.multi_channels = MLP(input_dim=self.input_dim*self.attn_output_dim,
                                  output_dim=self.attn_output_dim,
                                  hidden_dim=self.lin_hidden_dim, layer_num=1, activation='elu')
        self.final_layer = MLP(input_dim=2*self.input_dim, output_dim=self.output_dim,
                               hidden_dim=self.lin_hidden_dim, layer_num=self.lin_layer_num, activation='elu')
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, nodes_num_max, feature_input]
            adj: [batch_size, input_channels_dim, nodes_num_max, nodes_num_max]
        Returns: 
            x: [batch_size, nodes_num_max, feature_output]
            adj: [batch_size, output_channels_dim, nodes_num_max, nodes_num_max]
        """
        x_list, adj_list = [], []
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[1]), device=x.device)

        for i in range(self.input_dim):
            _x, _adj = self.attn[i](x, adj[:,i,:,:])
            x_list.append(_x)
            adj_list.append(_adj.unsqueeze(-1))
        x = self.multi_channels(torch.cat(x_list, dim=-1))
        x = x * mask[:,:,None]
        x = self.actv(x)

        adj = torch.cat([torch.cat(adj_list, dim=-1), adj.permute(0,2,3,1)], dim=-1)
        shape = adj.shape
        adj = self.final_layer(adj.view(-1,shape[-1]))
        adj = adj.view(shape[0],shape[1],shape[2],-1).permute(0,3,1,2)
        adj = adj + adj.transpose(-1,-2)

        mask = mask.unsqueeze(1)
        adj = adj * mask.unsqueeze(-1)
        adj = adj * mask.unsqueeze(-2)
        
        return x, adj

class DenseScoreNetworkX(nn.Module):
    def __init__(self, input_dim, hidden_dim: int=16, layer_num: int=2, actv=nn.Tanh()) -> None:
        super(DenseScoreNetworkX, self).__init__()
        """ Network for predict the score related to node features.
        Args:
            input_dim: dimension of input
            hidden_dim: dimension of hindden layer
            layer_num: number of hidden layer
            actv: type of activate function
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.actv = actv

        self.layers = torch.nn.ModuleList()
        for i in range(self.layer_num):
            if i==0:
                self.layers.append(DenseConv(self.input_dim, self.hidden_dim))
            else:
                self.layers.append(DenseConv(self.hidden_dim, self.hidden_dim))

        self.final_input_dim = self.input_dim + self.layer_num * self.hidden_dim
        self.final_layer = MLP(input_dim=self.final_input_dim, output_dim=self.input_dim,
                               hidden_dim=2*self.final_input_dim, layer_num=2, 
                               activation='elu')

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask=None) -> torch.Tensor:
        x_list = [x]
        for i in range(self.layer_num):
            x = self.layers[i](x, adj)
            x = self.actv(x)
            x_list.append(x)

        x = torch.cat(x_list, dim=-1) # [batch_size, nodes_num_max, (F + layer_num x H)]
        x_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final_layer(x).view(*x_shape)

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[1]), device=x.device)
        return x * mask[:,:,None]

class DenseScoreNetworkAdj(nn.Module):
    def __init__(self, input_dim: int=2, nodes_num_max: int=9, attn_input_dim: int=4,
                 attn_hidden_dim: int=16, attn_layer_num: int=3, attn_heads_num: int=4,
                 lin_output_dim: int=4, lin_hidden_dim: int=8, lin_layer_num: int=2,
                 conv_type='GCN') -> None:
        super(DenseScoreNetworkAdj, self).__init__()
        """ Network for predict the score related to node features.
        Args:
            input_dim: dimension of input
            nodes_num_max: maximum number of nodes in each graph
            attn_input_dim: dimension of input of attention layer
            attn_hidden_dim: dimension of hidden layer in attention layer
            attn_layer_num: number of attention layers
            attn_heads_num: number of heads in attention layer
            lin_output_dim: dimension of output of linear layer 
            lin_hidden_dim: dimension of hidden layer in linear layer
            lin_layer_num: number of linear layer
            conv_type: type of convolution layer
        """
        self.input_dim = input_dim
        self.nodes_num_max = nodes_num_max
        self.attn_input_dim = attn_input_dim
        self.attn_hidden_dim = attn_hidden_dim
        self.attn_layer_num = attn_layer_num
        self.attn_heads_num = attn_heads_num
        self.lin_output_dim = lin_output_dim
        self.lin_hidden_dim = lin_hidden_dim
        self.lin_layer_num = lin_layer_num
        self.conv_type = conv_type

        self.layers = torch.nn.ModuleList()
        for i in range(self.attn_layer_num):
            if i==0:
                self.layers.append(
                    DenseAttentionLayer(self.input_dim, self.lin_hidden_dim, self.lin_layer_num,
                                        self.attn_input_dim, self.attn_hidden_dim,
                                        self.attn_hidden_dim, self.attn_heads_num, self.conv_type))
            elif i==self.attn_layer_num-1:
                self.layers.append(
                    DenseAttentionLayer(self.lin_hidden_dim, self.lin_output_dim, self.lin_layer_num,
                                        self.attn_hidden_dim, self.attn_hidden_dim,
                                        self.attn_hidden_dim, self.attn_heads_num, self.conv_type))
            else:
                self.layers.append(
                    DenseAttentionLayer(self.lin_hidden_dim, self.lin_hidden_dim, self.lin_layer_num,
                                        self.attn_hidden_dim, self.attn_hidden_dim,
                                        self.attn_hidden_dim, self.attn_heads_num, self.conv_type))

        self.final_input_dim = self.lin_hidden_dim*(self.attn_layer_num-1) + self.lin_output_dim + self.input_dim
        self.final_layer = MLP(input_dim=self.final_input_dim, output_dim=1,
                               hidden_dim=2*self.final_input_dim, layer_num=2, 
                               activation='elu')
        
        self.mask = torch.ones([self.nodes_num_max, self.nodes_num_max]) - torch.eye(self.nodes_num_max)
        self.mask.unsqueeze_(0)

    def pow_tensor(self, x: torch.Tensor, cnum: torch.Tensor) -> torch.Tensor:
        _x = x.clone()
        xc = [x.unsqueeze(1)]
        for i in range(cnum-1):
            _x = torch.bmm(_x, x)
            xc.append(_x.unsqueeze(1))
        xc = torch.cat(xc, dim=1)
        return xc

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        adj = self.pow_tensor(adj, self.input_dim)
        adj_list = [adj]
        for i in range(self.attn_layer_num):
            x, adj = self.layers[i](x, adj, mask)
            adj_list.append(adj)
        
        adj = torch.cat(adj_list, dim=1).permute(0,2,3,1)
        score_shape = adj.shape[:-1]
        score = self.final_layer(adj).view(*score_shape)
        
        self.mask = self.mask.to(score.device)
        score = score * self.mask
        if mask is None:
            mask = torch.ones((score.shape[0], score.shape[1]), device=score.device)
        return score * mask[:,:,None]

def load_model_from_ckpt(params, state_dict, device):
    params_ = params.copy()
    model_type = params_.pop('model_type', None)
    
    if model_type == 'ScoreNetworkX':
        model = DenseScoreNetworkX(input_dim=params_['input_dim'],
            hidden_dim=params_['hidden_dim'], layer_num=params_['layer_num'])
        model.load_state_dict(state_dict)
        model.to(device)
    
    elif model_type == 'ScoreNetworkA':
        model = DenseScoreNetworkAdj(input_dim=params_['input_dim'],
            nodes_num_max=params_['nodes_num_max'], attn_input_dim=params_['attn_input_dim'],
            attn_hidden_dim=params_['attn_hidden_dim'], attn_layer_num=params_['attn_layer_num'],
            attn_heads_num=params_['attn_heads_num'], lin_output_dim=params_['lin_output_dim'], 
            lin_hidden_dim=params_['lin_hidden_dim'], lin_layer_num=params_['lin_layer_num'],
            conv_type=params_['conv_type'])
        model.load_state_dict(state_dict)
        model.to(device)
    
    return model
