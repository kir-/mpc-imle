import numpy as np
import jax.numpy as jnp
from functools import partial
import jax

#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#


def extract(a, t, x_shape):
    """
    Extract values from array 'a' at indices 't' and reshape.
    
    Args:
        a: [T] array
        t: [batch_size] indices
        x_shape: shape of x for reshaping
    
    Returns:
        Reshaped array [batch_size, 1, 1, ...]
    """
    b = t.shape[0]
    out = a[t]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=jnp.float32):
    """
    Cosine schedule for beta values.
    As proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    
    Args:
        timesteps: int - number of timesteps
        s: float - smoothing parameter
        dtype: JAX dtype
    
    Returns:
        jnp.ndarray: betas array of shape [timesteps]
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return jnp.array(betas_clipped, dtype=dtype)


def cond_dict_to_arrays(conditions):
    """
    conditions: dict {timestep(int): val [B, obs_dim]}
    Returns:
      ts:   [K] int32 timesteps
      vals: [K, B, obs_dim]
    """
    keys = tuple(sorted(conditions.keys()))
    ts = jnp.array(keys, dtype=jnp.int32)
    vals = jnp.stack([conditions[k] for k in keys], axis=0)
    return ts, vals


@partial(jax.jit, static_argnames=("action_dim",))
def apply_conditioning_packed(x, ts, vals, *, action_dim):
    ts = ts.astype(jnp.int32)
    vals_b = jnp.swapaxes(vals, 0, 1)  # [B, K, obs_dim]

    def set_for_one_batch(xb, vb):
        return xb.at[ts, action_dim:].set(vb)

    return jax.vmap(set_for_one_batch)(x, vals_b)

@jax.jit
def sort_by_values(x, values):
    """
    Sort trajectories by their corresponding values in descending order.
    
    Args:
        x: [batch_size, horizon, transition_dim] - trajectories
        values: [batch_size] - scalar values per trajectory
    
    Returns:
        tuple: (sorted_x, sorted_values)
    """
    inds = jnp.argsort(-values)
    return x[inds], values[inds]


def cond_to_tensor(cond):
    """
    Converts a conditioning dictionary to a single tensor.

    Args:
        cond (dict): Dictionary of arrays with shape [batch_size, feature_dim].

    Returns:
        jnp.ndarray: A tensor of shape [batch_size, total_features] obtained 
                     by concatenating values in sorted key order.
    """
    cond_tensor = jnp.stack([cond[key] for key in sorted(cond.keys())], axis=1)

    # Flatten all dimensions after batch dimension
    cond_tensor = cond_tensor.reshape(cond_tensor.shape[0], -1)

    return cond_tensor
