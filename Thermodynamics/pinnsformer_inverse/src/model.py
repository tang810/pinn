import torch
import torch.nn as nn
from .utils import get_clones

class WaveAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Tanh()  # 稳定激活函数

    def forward(self, x):
        return self.act(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__()
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
    PINNsformer for thermo-elastic problems (fully coupled)
    Input  : (x, y, z, t)
    Output : (T, u, v, w)
    """
    def __init__(self, d_model, d_hidden, N, heads, T0=273.0):
        super(PINNsformer, self).__init__()

        self.linear_emb = nn.Linear(4, d_model)

        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)

        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, 4)
        )

        # 可学习的输出偏置，初始化为给定值
        self.output_bias = nn.Parameter(
            torch.tensor([T0, 0., 0., 0.], dtype=torch.float32)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, y, z, t):
        src = torch.cat((x, y, z, t), dim=-1)
        src = self.linear_emb(src)
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        output = self.linear_out(d_output)
        output = output + self.output_bias
        return output