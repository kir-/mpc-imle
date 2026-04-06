import numpy as np
import torch

#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    t = t.to(a.device)
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x

def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values

def cond_to_tensor(cond):
        """
        Converts a conditioning dictionary to a single tensor.

        Args:
            cond (dict): Dictionary of tensors with shape [batch_size, feature_dim].

        Returns:
            torch.Tensor: A tensor of shape [batch_size, total_features] obtained by concatenating values in sorted key order.
        """
        cond_tensor = torch.stack([cond[key] for key in sorted(cond.keys())], dim=1)

        # Assert that the number of conditions is either 1 or 2
        assert cond_tensor.shape[1] in (1, 2), f"Expected 1 or 2 conditions, got {cond_tensor.shape[1]}"

        # If the second dimension is 2 (indicating start and goal state), flatten it
        if cond_tensor.shape[1] == 2:
            cond_tensor = cond_tensor.view(cond_tensor.size(0), -1)

        return cond_tensor