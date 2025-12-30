import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import trunc_normal_init_

# === RoPE ===
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=8192, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len, device):    # 动态截取
        if seq_len > self.cos_cached.shape[0]:
             pass 
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# === Linear with Trunc Init ===
class CastedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, input):
        return F.linear(input, self.weight.to(input.dtype), self.bias)

# === SwiGLU ===
class SwiGLU(nn.Module):
    def __init__(self, d_model, expansion=2.6):
        super().__init__()
        dim_hidden = int(d_model * expansion)
        self.gate_up = CastedLinear(d_model, dim_hidden * 2)
        self.down = CastedLinear(dim_hidden, d_model)
    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)

# === RMSNorm ===
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale