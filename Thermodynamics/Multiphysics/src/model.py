# implementation of PINNsformer
# paper: PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks
# link: https://arxiv.org/abs/2307.11833

import torch
import torch.nn as nn

def get_clones(module, N):
    import copy
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super(FeedForward, self).__init__()
        if d_ff is None:
            d_ff = d_model * 4
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            batch_first=True
        )
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x):
        x2 = self.act1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            batch_first=True
        )
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x, e_outputs):
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.act(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x, e_outputs):
        for layer in self.layers:
            x = layer(x, e_outputs)
        return self.act(x)


class PINNsformer(nn.Module):
    """
    PINNsformer for thermo-elastic problems
    
    Input  : (x, y, z, t)
    Output : (T)  # 改为单输出，只预测温度
    """
    def __init__(self, d_model, d_hidden, N, heads):
        super(PINNsformer, self).__init__()

        # ---- embedding ----
        self.linear_emb = nn.Linear(4, d_model)

        # ---- transformer ----
        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)

        # ---- output head ----
        # 只输出温度 T
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, 1)  # ← 改为1，只输出温度
        )

    def forward(self, x, y=None, z=None, t=None):
        """
        可以接受单独解包的张量，也可以直接接受一个包含四个特征的 x 矩阵 (兼容 evaluate.py)
        x, y, z, t : shape [N, 1] OR x: shape [batch, time, 4]
        
        return:
            output[:, 0] -> Temperature T
        """
        if y is None and z is None and t is None:
            src = x
        else:
            src = torch.cat((x, y, z, t), dim=-1)   # [N, 4]

        src = self.linear_emb(src)
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        output = self.linear_out(d_output)  # [N, 1]
        return output