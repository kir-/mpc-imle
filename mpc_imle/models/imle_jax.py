import os
from collections import namedtuple
from typing import Optional, Dict

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from dotenv import load_dotenv

from .helpers.losses_jax import Losses
from .helpers.sampling_jax import cond_to_tensor

load_dotenv()

DEVICE = os.getenv("DEVICE")

Sample = namedtuple("Sample", "trajectories values")


class IMLEModelJax(nn.Module):
    horizon: int
    observation_dim: int
    action_dim: int

    generator: nn.Module

    loss_type: str = "l2"
    sample_factor: int = 10
    noise_coef: float = 0.1
    latent_dim: int = 8

    action_weight: float = 1.0
    loss_discount: float = 1.0
    alpha: float = 0.0
    offset: float = 0.0
    loss_weights: Optional[Dict[int, float]] = None

    def setup(self):
        self.transition_dim = self.observation_dim + self.action_dim

        # build loss weights once (static for module instance)
        loss_weights = self._get_loss_weights(
            action_weight=self.action_weight,
            discount=self.loss_discount,
            weights_dict=self.loss_weights,
            dims=self.transition_dim,
        )
        self.loss_fn = Losses[self.loss_type](weights=loss_weights, action_dim=self.action_dim)

    def _get_loss_weights(
        self,
        action_weight: float,
        discount: float,
        weights_dict: Optional[Dict[int, float]],
        dims: int,
    ) -> jnp.ndarray:
        """Per-(t,dim) loss weights: [H, transition_dim]."""
        dim_weights = jnp.ones((dims,), dtype=jnp.float32)

        if weights_dict is None:
            weights_dict = {}

        # weights_dict keys are observation indices; shift by action_dim
        for ind, w in weights_dict.items():
            idx = self.action_dim + int(ind)
            dim_weights = dim_weights.at[idx].set(dim_weights[idx] * jnp.asarray(w, jnp.float32))

        discounts = (jnp.asarray(discount, jnp.float32) ** jnp.arange(self.horizon, dtype=jnp.float32))
        discounts = discounts / (discounts.mean() + 1e-8)

        # [H] x [D] -> [H, D]
        loss_weights = discounts[:, None] * dim_weights[None, :]

        # action weights at t=0
        loss_weights = loss_weights.at[0, : self.action_dim].set(action_weight)
        return loss_weights

    def conditional_sample(self, key: jax.Array, cond: dict, horizon: Optional[int] = None) -> Sample:
        """
        Pure JAX sampling. Guidance / ranking should be done OUTSIDE the module.
        """
        cond_tensor = cond_to_tensor(cond)
        batch_size = int(cond_tensor.shape[0])

        H = int(horizon or self.horizon)
        key, subkey = random.split(key)
        z = random.normal(subkey, (batch_size, H, self.latent_dim))

        trajectories = self.generator(z, cond_tensor)
        values = jnp.zeros((batch_size,), dtype=jnp.float32)
        return Sample(trajectories=trajectories, values=values)

    def __call__(self, key: jax.Array, cond: dict, horizon: Optional[int] = None) -> Sample:
        return self.conditional_sample(key, cond, horizon=horizon)

    def generate_latent(self, key: jax.Array, x: jnp.ndarray, cond: dict, rewards: jnp.ndarray) -> jnp.ndarray:
        """
        Called via: model.apply({"params": params}, key, x, cond, rewards, method=model.generate_latent)
        """
        B, H = int(x.shape[0]), int(x.shape[1])
        K = int(self.sample_factor)

        cond_tensor = cond_to_tensor(cond)

        key, subkey = random.split(key)
        zs = random.normal(subkey, (B * K, H, self.latent_dim))
        cond_rep = jnp.repeat(cond_tensor, K, axis=0)

        generated = self.generator(zs, cond_rep)  # [B*K, H, D]
        generated = generated.reshape(B, K, H, -1)

        # L2 over (time, dim)
        dists = jnp.linalg.norm(generated - x[:, None, :, :], axis=(2, 3))  # [B, K]
        nn_idx = jnp.argmin(dists, axis=1)  # [B]
        flat_idx = jnp.arange(B) * K + nn_idx
        z_nn = zs[flat_idx]  # [B, H, latent_dim]

        key, subkey = random.split(key)
        z_nn = z_nn + self.noise_coef * random.normal(subkey, z_nn.shape)

        return z_nn

    def loss(self, key: jax.Array, x: jnp.ndarray, cond: dict, rewards: jnp.ndarray, latent: jnp.ndarray, target=None):
        """
        Called via: model.apply({"params": params}, x, cond, rewards, latent, method=model.loss)
        """
        cond_tensor = cond_to_tensor(cond)
        outs = self.generator(latent, cond_tensor)

        reward_weights = self.offset + self.alpha + (1.0 - self.alpha) * rewards
        loss, infos = self.loss_fn(outs, x, reward_weights)
        return loss, infos
