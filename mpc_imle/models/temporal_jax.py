import jax
import jax.numpy as jnp
import flax.linen as nn

from .helpers.modules_jax import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)

def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))

class Identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x

class ResidualTemporalBlock(nn.Module):
    inp_channels: int
    out_channels: int
    embed_dim: int
    kernel_size: int = 5

    def setup(self):
        # main blocks
        self.block1 = Conv1dBlock(self.inp_channels, self.out_channels, self.kernel_size)
        self.block2 = Conv1dBlock(self.out_channels, self.out_channels, self.kernel_size)

        # time projection
        self.time_dense = nn.Dense(self.out_channels)

        # residual projection (optional)
        self.res_conv = None
        if self.inp_channels != self.out_channels:
            self.res_conv = nn.Conv(
                features=self.out_channels,
                kernel_size=(1,),
                padding="VALID",
            )
            
    def __call__(self, x, t):
        """
        x: [B, horizon, inp_channels]
        t: [B, embed_dim]
        returns: [B, out_channels, horizon]
        """
        # main blocks
        h = self.block1(x)

        # time mlp: Mish -> Linear(embed_dim -> out_channels) -> [B, C, 1]
        time = mish(t)
        time = self.time_dense(time)      # [B, C]
        time = time[:, None, :]   # [B, 1, C] (broadcast over T)


        h = h + time
        h = self.block2(h)

        residual = self.res_conv(x) if self.res_conv is not None else x
        return h + residual

class TemporalUnetJax(nn.Module):
    horizon: int
    transition_dim: int
    cond_dim: int
    dim: int = 32
    dim_mults: tuple = (1, 2, 4, 8)
    attention: bool = False

    def setup(self):
        dims = [self.transition_dim, *[self.dim * m for m in self.dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.in_out = tuple(in_out)
        num_resolutions = len(in_out)

        time_dim = self.dim
        self.time_mlp = nn.Sequential([
            SinusoidalPosEmb(self.dim),
            nn.Dense(self.dim * 4),
            mish,
            nn.Dense(self.dim),
        ])

        # downs
        downs = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            block1 = ResidualTemporalBlock(
                inp_channels=dim_in,
                out_channels=dim_out,
                embed_dim=time_dim,
            )
            block2 = ResidualTemporalBlock(
                inp_channels=dim_out,
                out_channels=dim_out,
                embed_dim=time_dim,
            )
            attn = (
                Residual(fn=PreNorm(dim=dim_out, fn=LinearAttention(dim=dim_out)))
                if self.attention else Identity()
            )
            down = Downsample1d(dim=dim_out) if not is_last else Identity()

            downs.append((block1, block2, attn, down))

        self.downs = tuple(downs)

        # middle
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_attn   = (Residual(fn=PreNorm(dim=mid_dim, fn=LinearAttention(dim=mid_dim)))
                           if self.attention else Identity())
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)

        # ups
        ups = []
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # torch uses dim=1 cat (channels). We'll do axis=1 since [B,C,T].
            is_last = ind >= (num_resolutions - 1)

            block1 = ResidualTemporalBlock(
                inp_channels=dim_out * 2,
                out_channels=dim_in,
                embed_dim=time_dim,
            )
            block2 = ResidualTemporalBlock(
                inp_channels=dim_in,
                out_channels=dim_in,
                embed_dim=time_dim,
            )
            attn = (
                Residual(fn=PreNorm(dim=dim_in, fn=LinearAttention(dim=dim_in)))
                if self.attention else Identity()
            )
            up = Upsample1d(dim=dim_in) if not is_last else Identity()

            ups.append((block1, block2, attn, up))

        self.ups = tuple(ups)

        self.final_conv = nn.Sequential([
            Conv1dBlock(inp_channels=self.dim, out_channels=self.dim, kernel_size=5),
            nn.Conv(features=self.transition_dim, kernel_size=(1,), padding="VALID"),
        ])

    def __call__(self, x, cond, time):

        t = self.time_mlp(time)   # [B, dim]
        h_list = []

        for res1, res2, attn, down in self.downs:
            x = res1(x, t)
            x = res2(x, t)
            x = attn(x)
            h_list.append(x)
            x = down(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for res1, res2, attn, up in self.ups:
            x = jnp.concatenate([x, h_list.pop()], axis=-1)
            x = res1(x, t)
            x = res2(x, t)
            x = attn(x)
            x = up(x)

        x = self.final_conv(x)         # [B, D, H]
        return x


class ValueFunctionJax(nn.Module):
    horizon: int
    transition_dim: int
    cond_dim: int
    dim: int = 32
    dim_mults: tuple = (1, 2, 4, 8)
    out_dim: int = 1

    def setup(self):
        dims = [self.transition_dim, *[self.dim * m for m in self.dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        time_dim = self.dim
        self.time_mlp = nn.Sequential([
            SinusoidalPosEmb(self.dim),
            nn.Dense(self.dim * 4),
            mish,
            nn.Dense(self.dim),
        ])

        blocks = []
        horizon = self.horizon
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            b1 = ResidualTemporalBlock(dim_in,  dim_out, embed_dim=time_dim, kernel_size=5)
            b2 = ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, kernel_size=5)
            down = Downsample1d(dim=dim_out)

            blocks.append((b1, b2, down))
            if not is_last:
                horizon //= 2

        self.blocks = tuple(blocks)

        mid_dim   = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4

        self.mid_block1 = ResidualTemporalBlock(mid_dim,   mid_dim_2, embed_dim=time_dim, kernel_size=5)
        self.mid_down1  = Downsample1d(dim=mid_dim_2)
        horizon //= 2

        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, embed_dim=time_dim, kernel_size=5)
        self.mid_down2  = Downsample1d(dim=mid_dim_3)
        horizon //= 2

        self.fc_dim = mid_dim_3 * max(horizon, 1)

        self.fc1 = nn.Dense(self.fc_dim // 2)
        self.fc2 = nn.Dense(self.out_dim)

    def __call__(self, x, cond, time, *args):
        """
        x: [B, H, transition_dim]
        returns: [B, out_dim]
        """

        t = self.time_mlp(time)  # [B, time_dim]

        for b1, b2, down in self.blocks:
            x = b1(x, t)
            x = b2(x, t)
            x = down(x)

        x = self.mid_block1(x, t)
        x = self.mid_down1(x)
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)

        x = x.reshape((x.shape[0], -1))            # [B, fc_dim]
        xt = jnp.concatenate([x, t], axis=-1)      # [B, fc_dim + time_dim]

        h = mish(self.fc1(xt))
        out = self.fc2(h)
        return out
