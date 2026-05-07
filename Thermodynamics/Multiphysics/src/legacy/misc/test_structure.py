import torch
import torch.nn as nn

class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
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
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + self.act1(attn_out)
        ff_out = self.ff(x)
        x = x + self.act2(ff_out)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        # Note: only ONE attention!
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x, enc_out):
        attn_out, _ = self.attn(x, enc_out, enc_out)
        x = x + self.act1(attn_out)
        ff_out = self.ff(x)
        x = x + self.act2(ff_out)
        return x

class EncoderWrapper(nn.Module):
    def __init__(self, d_model, N, heads):
        super(EncoderWrapper, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(N)])
        self.act = WaveAct()
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.act(x)

class DecoderWrapper(nn.Module):
    def __init__(self, d_model, N, heads):
        super(DecoderWrapper, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(N)])
        self.act = WaveAct()
    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.act(x)

class PINNsformerStr(nn.Module):
    def __init__(self, d_model=64, d_hidden=512, N=1, heads=2):
        super(PINNsformerStr, self).__init__()
        self.linear_emb = nn.Linear(4, d_model)
        self.encoder = EncoderWrapper(d_model, N, heads)
        self.decoder = DecoderWrapper(d_model, N, heads)
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, 1)
        )

    def forward(self, x):
        src = self.linear_emb(x)
        enc_out = self.encoder(src)
        dec_out = self.decoder(src, enc_out)
        out = self.linear_out(dec_out)
        return out

if __name__ == '__main__':
    model = PINNsformerStr()
    state = torch.load('model/pinnsformer_withsun.pt', map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=True)
    print("Missing:", missing)
    print("Unexpected:", unexpected)
    print("SUCCESS")
