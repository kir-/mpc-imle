import torch
import torch.nn as nn
import einops

from .helpers.modules import (
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
    AdaptiveGroupNorm,
    AdaptiveLayerNorm
)

class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, cond_dim, kernel_size=5, use_layer_norm=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        # Adaptive normalization layers
        if (use_layer_norm):
            # Layer Norm
            self.norm1 = AdaptiveLayerNorm(out_channels, cond_dim)
            self.norm2 = AdaptiveLayerNorm(out_channels, cond_dim)
        else:
            # Group Norm
            self.norm1 = AdaptiveGroupNorm(out_channels, cond_dim)
            self.norm2 = AdaptiveGroupNorm(out_channels, cond_dim) 
        
        # Residual connection
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        x    : [batch_size x inp_channels x horizon]
        cond : [batch_size x cond_dim]
        """
        # Process the input through the convolutional blocks with conditional normalization
        h = self.blocks[0](x)
        h = self.norm1(h, cond)
        
        h = self.blocks[1](h)
        h = self.norm2(h, cond)

        # Add residual connection
        return h + self.residual_conv(x)

class TemporalUnetIMLE(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        latent_dim=8,
        dim_mults=(1, 2, 4, 8),
        attention=False,
        use_layer_norm=False
    ):
        super().__init__()

        self.horizon = horizon
        dims = [latent_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, cond_dim=cond_dim, use_layer_norm=use_layer_norm),
                ResidualTemporalBlock(dim_out, dim_out, cond_dim=cond_dim, use_layer_norm=use_layer_norm),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim=cond_dim, use_layer_norm=use_layer_norm)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim=cond_dim, use_layer_norm=use_layer_norm)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, cond_dim=cond_dim, use_layer_norm=use_layer_norm),
                ResidualTemporalBlock(dim_in, dim_in, cond_dim=cond_dim, use_layer_norm=use_layer_norm),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, latents, cond):
        """
        latents    : [batch x latent_dim x horizon ]
        cond : [batch x cond_dim]
        """

        x = latents
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1) 
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t') 
        return x
