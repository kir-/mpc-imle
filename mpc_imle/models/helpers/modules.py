import math
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
    
class AdaptiveMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim * 2)  # Outputs scale and shift
        )

    def forward(self, x):
        return self.net(x)

#-----------------------------------------------------------------------------#
#--------------------------------- attention ---------------------------------#
#-----------------------------------------------------------------------------#

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
class AdaptiveLayerNorm(nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.cond_proj = AdaptiveMLP(cond_dim, num_features)

    def forward(self, x, cond):
        x = x.permute(0, 2, 1)
        scale_shift = self.cond_proj(cond)  # [B, 2*D]
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)  # [B, D], [B, D]

        normalized_x = self.norm(x) 

        normalized_x =  normalized_x * (1 + scale) + shift
        return normalized_x.permute(0, 2, 1)  # back to [B, D, T]
    
class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_channels, cond_dim):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, num_channels), num_channels)  # Standard GroupNorm
        
        # Add condition projection directly in the norm class
        self.cond_proj = AdaptiveMLP(cond_dim, num_channels)

    def forward(self, x, cond):
        # Generate scale and shift from condition
        scale_shift = self.cond_proj(cond)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        
        # Apply normalization and conditioning
        normalized_x = self.norm(x)
        
        # Reshape scale and shift for broadcasting
        scale = scale.view(scale.shape[0], -1, *([1] * (normalized_x.dim() - 2)))
        shift = shift.view(shift.shape[0], -1, *([1] * (normalized_x.dim() - 2)))
        
        return normalized_x * (1 + scale) + shift

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)