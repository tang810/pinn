import torch
import numpy as np
import networkx as nx

class NodeTypeMask():
    def __init__(self, node_type_list, mask_list):
        self.node_type_list = node_type_list
        self.mask_list = mask_list
    
    def __call__(self, node_type):
        mask = torch.zeros(node_type.shape, dtype=torch.bool)
        for k in self.mask_list:
            mask |= (node_type==self.node_type_list[k])
        return mask

def pad_adj(ori_adj, node_number):
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])], axis=0)
    return a

def graphs_to_tensor(graph_list, max_node_num):
    adj_list = []
    max_node_num = max_node_num

    for g in graph_list:
        assert isinstance(g, nx.Graph)
        node_list = []
        for v, feature in g.nodes.data('feature'):
            node_list.append(v)

        adj = nx.to_numpy_array(g, nodelist=node_list)
        padded_adj = pad_adj(adj, node_number=max_node_num)
        adj_list.append(padded_adj)

    adj_np = np.asarray(adj_list)
    adj_tensor = torch.tensor(adj_np, dtype=torch.float32)
    del graph_list
    del adj_list
    del adj_np

    return adj_tensor 

def node_mask(adj, eps=1e-5):
    mask = torch.abs(adj).sum(-1).gt(eps).to(dtype=torch.float32)
    if len(mask.shape)==3:
        mask = mask[:,0,:]
    return mask

def init_mask(graph_list, config, batch_size=None):
    if batch_size is None:
        batch_size = config.data.batch_size
    max_node_num = config.data.max_node_num
    graph_tensor = graphs_to_tensor(graph_list, max_node_num)
    idx = np.random.randint(0, len(graph_list), batch_size)
    mask = node_mask(graph_tensor[idx])

    return mask

def mask_x(x, mask):
    """ Mask batch of node features with 0-1 tensor
    """
    if mask is None:
        mask = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * mask[:,:,None]

def mask_adj(adj, mask):
    """ Mask batch of adjacency matrices with 0-1 mask tensor
    Args:
        adj: [batch_size,node_size,node_size] or [batch_size,channel_size,node_size,node_size]
        mask: [batch_size,node_size]
    """
    if mask is None:
        mask = torch.ones((adj.shape[0], adj.shape[-1]), device=adj.device)
    if len(adj.shape) == 4:
        mask = mask.unsqueeze(1)  # [batch_size,1,node_size]
    adj = adj * mask.unsqueeze(-1)
    adj = adj * mask.unsqueeze(-2)
    return adj

def quantize_qm9(x, adj):                         
    if type(adj).__name__ == 'Tensor':
        adj = adj.detach().cpu()
    else:
        adj = torch.tensor(adj)
    adj[adj >= 2.5] = 3
    adj[torch.bitwise_and(adj >= 1.5, adj < 2.5)] = 2
    adj[torch.bitwise_and(adj >= 0.5, adj < 1.5)] = 1
    adj[adj < 0.5] = 0
    
    adj = np.array(adj.to(torch.int64))
    adj -= 1
    adj[adj==-1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2
    adj = torch.nn.functional.one_hot(torch.tensor(adj), num_classes=4).permute(0, 3, 1, 2)
    
    x = torch.where(x > 0.5, 1, 0)
    x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)      # 32, 9, 4 -> 32, 9, 5
    return x, adj
