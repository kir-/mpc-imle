import jax.numpy as jnp
import flax.linen as nn

from .helpers.modules_jax import (
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
    AdaptiveGroupNorm,
    AdaptiveLayerNorm,
)

class Identity(nn.Module):
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        return x


class ResidualTemporalBlock(nn.Module):
    inp_channels: int
    out_channels: int
    cond_dim: int
    kernel_size: int = 5
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, cond):
        """
        x:    [B, T, inp_channels]
        cond: [B, cond_dim]
        """
        # First conv block
        h = Conv1dBlock(
            inp_channels=self.inp_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
        )(x)

        # First adaptive norm
        if self.use_layer_norm:
            h = AdaptiveLayerNorm(
                num_features=self.out_channels,
                cond_dim=self.cond_dim,
            )(h, cond)
        else:
            h = AdaptiveGroupNorm(
                num_channels=self.out_channels,
                cond_dim=self.cond_dim,
            )(h, cond)

        # Second conv block
        h = Conv1dBlock(
            inp_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
        )(h)

        # Second adaptive norm
        if self.use_layer_norm:
            h = AdaptiveLayerNorm(
                num_features=self.out_channels,
                cond_dim=self.cond_dim,
            )(h, cond)
        else:
            h = AdaptiveGroupNorm(
                num_channels=self.out_channels,
                cond_dim=self.cond_dim,
            )(h, cond)

        if self.inp_channels != self.out_channels:
            residual = nn.Conv(
                features=self.out_channels,
                kernel_size=(1,),
                padding='SAME',
            )(x)
        else:
            residual = x

        return h + residual


class TemporalUnetIMLEJax(nn.Module):
    horizon: int
    transition_dim: int
    cond_dim: int
    dim: int = 32
    latent_dim: int = 8
    dim_mults: tuple = (1, 2, 4, 8)
    attention: bool = False
    use_layer_norm: bool = False

    def setup(self):
        # dims = [latent_dim, dim*1, dim*2, dim*4, dim*8]
        dims = [self.latent_dim, *[self.dim * m for m in self.dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))  # [(ldim, d1), (d1, d2), ...]

        downs = []
        num_resolutions = len(in_out)
        horizon = self.horizon

        # Down path
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            block1 = ResidualTemporalBlock(
                inp_channels=dim_in,
                out_channels=dim_out,
                cond_dim=self.cond_dim,
                use_layer_norm=self.use_layer_norm,
            )
            block2 = ResidualTemporalBlock(
                inp_channels=dim_out,
                out_channels=dim_out,
                cond_dim=self.cond_dim,
                use_layer_norm=self.use_layer_norm,
            )
            attn = (
                Residual(
                    fn=PreNorm(dim=dim_out, fn=LinearAttention(dim=dim_out))
                )
                if self.attention
                else Identity()
            )
            downsample = Downsample1d(dim=dim_out) if not is_last else Identity()

            downs.append((block1, block2, attn, downsample))

            if not is_last:
                horizon //= 2 

        self.downs = tuple(downs)

        # Middle blocks
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            inp_channels=mid_dim,
            out_channels=mid_dim,
            cond_dim=self.cond_dim,
            use_layer_norm=self.use_layer_norm,
        )
        self.mid_attn = (
            Residual(
                fn=PreNorm(dim=mid_dim, fn=LinearAttention(dim=mid_dim))
            )
            if self.attention
            else Identity()
        )
        self.mid_block2 = ResidualTemporalBlock(
            inp_channels=mid_dim,
            out_channels=mid_dim,
            cond_dim=self.cond_dim,
            use_layer_norm=self.use_layer_norm,
        )

        ups = []
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            block1 = ResidualTemporalBlock(
                inp_channels=dim_out * 2,
                out_channels=dim_in,
                cond_dim=self.cond_dim,
                use_layer_norm=self.use_layer_norm,
            )
            block2 = ResidualTemporalBlock(
                inp_channels=dim_in,
                out_channels=dim_in,
                cond_dim=self.cond_dim,
                use_layer_norm=self.use_layer_norm,
            )
            attn = (
                Residual(
                    fn=PreNorm(dim=dim_in, fn=LinearAttention(dim=dim_in))
                )
                if self.attention
                else Identity()
            )
            upsample = Upsample1d(dim=dim_in) if not is_last else Identity()

            ups.append((block1, block2, attn, upsample))

            if not is_last:
                horizon *= 2

        self.ups = tuple(ups)

        # Conv1dBlock(dim, dim, 5) -> Conv1d(dim, transition_dim, 1)
        self.final_conv1 = Conv1dBlock(
            inp_channels=self.dim,
            out_channels=self.dim,
            kernel_size=5,
        )
        self.final_conv2 = nn.Conv(
            features=self.transition_dim,
            kernel_size=(1,),
        )

    def __call__(self, latents, cond):
        """
        latents: [B, T=horizon, latent_dim]
        cond:    [B, cond_dim]
        returns: [B, T=horizon, transition_dim]
        """
        x = latents
        hs = []

        # Down path
        for res1, res2, attn, down in self.downs:
            x = res1(x, cond)
            x = res2(x, cond)
            x = attn(x)
            hs.append(x)
            x = down(x)

        # Middle
        x = self.mid_block1(x, cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond)

        # Up path
        for res1, res2, attn, up in self.ups:
            skip = hs.pop()
            x = jnp.concatenate([x, skip], axis=-1)  # concat on channels
            x = res1(x, cond)
            x = res2(x, cond)
            x = attn(x)
            x = up(x)

        x = self.final_conv1(x)
        x = self.final_conv2(x)  # [B, T, transition_dim]
        return x
