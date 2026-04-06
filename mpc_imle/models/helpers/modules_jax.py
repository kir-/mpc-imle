import math
import jax
import jax.numpy as jnp
from flax import linen as nn
import einops

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        """
        x: [B] or [B, 1] timesteps (float or int)
        returns: [B, dim]
        """
        x = jnp.asarray(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]

        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = jnp.exp(jnp.arange(half_dim) * -emb_scale)  # [half_dim]
        # broadcast: [B] -> [B, 1], [half_dim] -> [1, half_dim]
        args = x[:, None] * freqs[None, :]
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        return emb  # [B, dim]


class Downsample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        """
        x: [B, T, C=dim]
        Conv1d(dim, dim, 3, stride=2, padding=1) equivalent
        """
        x = nn.Conv(
            features=self.dim,
            kernel_size=(3,),
            strides=(2,),
            padding='SAME',   # close to PyTorch padding=1
        )(x)  # [B, T//2, dim]
        return x


class Upsample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        """
        x: [B, T, C=dim]
        Approximate ConvTranspose1d(dim, dim, 4, 2, 1):
        - nearest-neighbor upsample by 2 in time
        - 1x3 conv for smoothing
        """
        if x.ndim != 3:
            raise ValueError(f"Upsample1d expected x.ndim==3 [B,T,C], got shape {x.shape}")
        B, T, C = x.shape
        # upsample time dimension by 2
        x = jax.image.resize(x, shape=(B, T * 2, C), method="nearest")
        # optional conv for learnable upsampling
        x = nn.Conv(
            features=self.dim,
            kernel_size=(3,),
            padding='SAME',
        )(x)  # [B, 2T, dim]
        return x


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    PyTorch version expects [B, C, T]; here we use [B, T, C].
    """
    inp_channels: int
    out_channels: int
    kernel_size: int
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        """
        x: [B, T, inp_channels]
        """
        # Conv1d over time axis
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            padding='SAME',
        )(x)  # [B, T, out_channels]

        # GroupNorm over channel axis (last); older flax GroupNorm has no `axis` kwarg
        x = nn.GroupNorm(
            num_groups=self.n_groups,
        )(x)  # [B, T, out_channels]

        # Mish: x * tanh(softplus(x))
        x = x * jnp.tanh(jax.nn.softplus(x))
        return x


class AdaptiveMLP(nn.Module):
    input_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        """
        x: [B, input_dim]
        returns: [B, 2 * output_dim]   (scale, shift)
        """
        x = nn.Dense(self.output_dim)(x)
        x = jax.nn.silu(x)
        x = nn.Dense(self.output_dim)(x)
        x = jax.nn.silu(x)
        x = nn.Dense(self.output_dim * 2)(x)
        return x


#-----------------------------------------------------------------------------#
#--------------------------------- attention ---------------------------------#
#-----------------------------------------------------------------------------#

class Residual(nn.Module):
    fn: nn.Module

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    """
    Custom LN that mirrors the PyTorch one, but on [B, T, C] and normalizing over C.
    """
    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        """
        x: [B, T, C=dim]
        """
        g = self.param('g', nn.initializers.ones, (1, 1, self.dim))
        b = self.param('b', nn.initializers.zeros, (1, 1, self.dim))

        mean = jnp.mean(x, axis=-1, keepdims=True)  # over channels
        var = jnp.var(x, axis=-1, keepdims=True)
        x_hat = (x - mean) / jnp.sqrt(var + self.eps)
        return x_hat * g + b


class PreNorm(nn.Module):
    dim: int
    fn: nn.Module

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = LayerNorm(self.dim)(x)
        return self.fn(x, *args, **kwargs)


class AdaptiveLayerNorm(nn.Module):
    """
    PyTorch:
      - permute [B, C, T] -> [B, T, C]
      - LayerNorm over C
      - scale/shift from cond
      - permute back
    Here we assume input already [B, T, C].
    """
    num_features: int
    cond_dim: int

    @nn.compact
    def __call__(self, x, cond):
        """
        x:    [B, T, C=num_features]
        cond: [B, cond_dim]
        """
        # Project cond -> scale, shift
        scale_shift = AdaptiveMLP(self.cond_dim, self.num_features)(cond)  # [B, 2C]
        scale, shift = jnp.split(scale_shift, 2, axis=-1)                  # [B, C], [B, C]
        scale = scale[:, None, :]  # [B, 1, C]
        shift = shift[:, None, :]  # [B, 1, C]

        # LayerNorm over channels
        x_norm = nn.LayerNorm()(x)  # [B, T, C]

        return x_norm * (1.0 + scale) + shift


class AdaptiveGroupNorm(nn.Module):
    num_channels: int
    cond_dim: int

    @nn.compact
    def __call__(self, x, cond):
        """
        x:    [B, T, C=num_channels]
        cond: [B, cond_dim]
        """
        # GroupNorm over channel axis (last)
        x_norm = nn.GroupNorm(
            num_groups=min(32, self.num_channels),
        )(x)  # [B, T, C]

        # cond -> scale, shift
        scale_shift = AdaptiveMLP(self.cond_dim, self.num_channels)(cond)  # [B, 2C]
        scale, shift = jnp.split(scale_shift, 2, axis=-1)                  # [B, C], [B, C]

        # expand for broadcasting over time
        # [B, C] -> [B, 1, C]
        scale = scale[:, None, :]
        shift = shift[:, None, :]

        return x_norm * (1.0 + scale) + shift


class LinearAttention(nn.Module):
    """
    Linear attention in 1D over time, channels-last [B, T, C].

    PyTorch:
      - Conv1d(dim -> 3 * hidden_dim, kernel=1) on [B, C, T]
      - chunk along C
      - softmax over time
    Here we use Dense over channels, which is equivalent to a 1x1 conv.
    """
    dim: int
    heads: int = 4
    dim_head: int = 32

    @nn.compact
    def __call__(self, x):
        """
        x: [B, T, C=dim]
        returns: [B, T, C]
        """
        scale = self.dim_head ** -0.5
        hidden_dim = self.heads * self.dim_head

        # project to q, k, v
        qkv = nn.Dense(3 * hidden_dim, use_bias=False)(x)  # [B, T, 3H]
        q, k, v = jnp.split(qkv, 3, axis=-1)               # each [B, T, H]

        # reshape to heads
        q = einops.rearrange(q, 'b t (h c) -> b h t c', h=self.heads)  # [B, H, T, C_h]
        k = einops.rearrange(k, 'b t (h c) -> b h t c', h=self.heads)
        v = einops.rearrange(v, 'b t (h c) -> b h t c', h=self.heads)

        # move to [B, H, C_h, T] to match einsum pattern
        q = einops.rearrange(q, 'b h t c -> b h c t')  # [B, H, C, T]
        k = einops.rearrange(k, 'b h t c -> b h c t')
        v = einops.rearrange(v, 'b h t c -> b h c t')

        q = q * scale

        # softmax over time dimension (n)
        k = jax.nn.softmax(k, axis=-1)  # [B, H, C, T]

        # context: [B, H, C_q, C_v]
        context = jnp.einsum('b h d n, b h e n -> b h d e', k, v)

        # out: [B, H, C_out, T]
        out = jnp.einsum('b h d e, b h d n -> b h e n', context, q)

        # back to [B, T, hidden_dim]
        out = einops.rearrange(out, 'b h c n -> b n (h c)')

        # final projection back to dim
        out = nn.Dense(self.dim)(out)  # [B, T, dim]
        return out
