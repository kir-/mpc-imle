from collections import namedtuple
from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from .helpers.losses_jax import Losses, WeightedLoss
from .helpers.sampling_jax import (
    cosine_beta_schedule,
    extract,
    cond_dict_to_arrays,
    apply_conditioning_packed,
    sort_by_values,
    cond_to_tensor,
)

Sample = namedtuple("Sample", "trajectories values chains")

class _ScanStep(nn.Module):
    diffusion: Any
    cond_tensor: Any
    cond_packed: Any

    @nn.compact
    def __call__(self, carry, i):
        key, x = carry
        t = jnp.full((x.shape[0],), i, dtype=jnp.int32)
        key, x, values = self.diffusion.p_sample_step(key, x, self.cond_tensor, self.cond_packed, t)
        return (key, x), (x, values)


class GaussianDiffusionJax(nn.Module):
    """
    JIT-first diffusion module:
      - __call__(key, cond, horizon=None, return_chain=False) -> Sample
      - loss(key, x, cond, ...) -> (loss, info, key)
    """
    horizon: int
    observation_dim: int
    action_dim: int
    generator: nn.Module

    n_timesteps: int = 1000
    loss_type: str = "l1"
    clip_denoised: bool = False
    predict_epsilon: bool = True

    action_weight: float = 1.0
    loss_discount: float = 1.0
    loss_weights: Optional[Dict[int, float]] = None
    latent_dim: int = 1

    def setup(self):
        self.transition_dim = self.observation_dim + self.action_dim
        T = int(self.n_timesteps)

        betas = cosine_beta_schedule(T).astype(jnp.float32)  # [T]
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.concatenate(
            [jnp.ones((1,), dtype=jnp.float32), alphas_cumprod[:-1]],
            axis=0,
        )

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # q(x_t | x_0)
        self.sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod - 1.0)

        # q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = jnp.log(jnp.clip(posterior_variance, a_min=1e-20))
        self.posterior_mean_coef1 = betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cumprod)

        loss_w = self._get_loss_weights(
            action_weight=self.action_weight,
            discount=self.loss_discount,
            weights_dict=self.loss_weights,
            dims=self.transition_dim,
        )
        loss_cls = Losses[self.loss_type]

        if issubclass(loss_cls, WeightedLoss):
            self.loss_fn = loss_cls(weights=loss_w, action_dim=self.action_dim)
        else:
            self.loss_fn = loss_cls()


    def _get_loss_weights(
        self,
        action_weight: float,
        discount: float,
        weights_dict: Optional[Dict[int, float]],
        dims: int,
    ) -> jnp.ndarray:
        dim_weights = jnp.ones((dims,), dtype=jnp.float32)
        if weights_dict is None:
            weights_dict = {}

        for ind, w in weights_dict.items():
            idx = self.action_dim + int(ind)
            dim_weights = dim_weights.at[idx].set(dim_weights[idx] * jnp.asarray(w, jnp.float32))

        discounts = (jnp.asarray(discount, jnp.float32) ** jnp.arange(self.horizon, dtype=jnp.float32))
        discounts = discounts / (discounts.mean() + 1e-8)

        loss_weights = discounts[:, None] * dim_weights[None, :]  # [H, D]
        loss_weights = loss_weights.at[0, : self.action_dim].set(action_weight)
        return loss_weights

    # ------------------------- diffusion math ------------------------- #

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise  # model predicts x0 directly

    def q_posterior(self, x_start, x_t, t):
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def p_mean_variance(self, x, cond_tensor, t):
        model_out = self.generator(x, cond_tensor, t)
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)

        if self.clip_denoised:
            x_recon = jnp.clip(x_recon, -1.0, 1.0)

        model_mean, posterior_var, posterior_log_var = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_var, posterior_log_var

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ------------------------- sampling (jit/scan only) ------------------------- #

    def _apply_cond_packed(self, x: jnp.ndarray, cond_packed):
        ts, vals = cond_packed
        return apply_conditioning_packed(x, ts, vals, action_dim=self.action_dim)

    def p_sample_step(self, key, x, cond_tensor, cond_packed, t):
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond_tensor=cond_tensor, t=t)
        model_std = jnp.exp(0.5 * model_log_variance)

        key, sub = random.split(key)
        noise = random.normal(sub, x.shape, dtype=x.dtype)

        mask = (t != 0).astype(x.dtype).reshape((x.shape[0],) + (1,) * (x.ndim - 1))
        x = model_mean + model_std * (noise * mask)

        x = self._apply_cond_packed(x, cond_packed)
        values = jnp.zeros((x.shape[0],), dtype=jnp.float32)
        return key, x, values
    
    def generate_latent(self, key, x, cond, rewards):
        z = jnp.zeros((x.shape[0], x.shape[1], 1), dtype=jnp.float32)
        return z

    @nn.compact
    def p_sample_loop_scan(self, key, shape, cond_tensor, cond_packed, return_chain: bool = False):
        key, sub = random.split(key)
        x = random.normal(sub, shape, dtype=jnp.float32)
        x = self._apply_cond_packed(x, cond_packed)

        timesteps = jnp.arange(self.n_timesteps - 1, -1, -1, dtype=jnp.int32)

        ScanStep = nn.scan(
            _ScanStep,
            variable_broadcast="params",
            split_rngs={"params": False, "dropout": True},
            in_axes=0,      # only timesteps (the i argument) is scanned
            out_axes=0,
        )

        scanned = ScanStep(diffusion=self, cond_tensor=cond_tensor, cond_packed=cond_packed)
        (key, x_final), (xs, values_hist) = scanned((key, x), timesteps)

        values = values_hist[-1]  # all zeros unless you add guidance/ranking inside scan

        chain = jnp.swapaxes(xs, 0, 1) if return_chain else None  # [B, T, H, D]
        x_sorted, v_sorted = sort_by_values(x_final, values)
        return Sample(trajectories=x_sorted, values=v_sorted, chains=chain), key

    def __call__(self, key: jax.Array, cond: dict, horizon: Optional[int] = None, return_chain: bool = False) -> Sample:

        cond_tensor = cond_to_tensor(cond)
        ts, vals = cond_dict_to_arrays(cond)

        B = int(cond_tensor.shape[0])
        H = int(horizon or self.horizon)
        shape = (B, H, self.transition_dim)

        sample, _ = self.p_sample_loop_scan(key, shape, cond_tensor, (ts, vals), return_chain=return_chain)
        return sample

    # ------------------------- training ------------------------- #

    def p_losses(self, key: jax.Array, x_start: jnp.ndarray, cond: dict, t: jnp.ndarray):
        cond_tensor = cond_to_tensor(cond)
        ts, vals = cond_dict_to_arrays(cond)
        cond_packed = (ts, vals)

        key, sub = random.split(key)
        noise = random.normal(sub, x_start.shape, dtype=x_start.dtype)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if cond_packed is not None:
            x_noisy = self._apply_cond_packed(x_noisy, cond_packed)

        x_recon = self.generator(x_noisy, cond_tensor, t)
        if cond_packed is not None:
            x_recon = self._apply_cond_packed(x_recon, cond_packed)

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, key: jax.Array, x: jnp.ndarray, cond: dict, rewards=None,latent=None, target=None):
        B = int(x.shape[0])
        key, sub = random.split(key)
        t = random.randint(sub, (B,), minval=0, maxval=int(self.n_timesteps), dtype=jnp.int32)
        return self.p_losses(key, x, cond, t)


class ValueDiffusionJax(GaussianDiffusionJax):
    """
    ValueDiffusion: model predicts a target (e.g., value), not epsilon.
    """

    def p_losses(self, key: jax.Array, x_start: jnp.ndarray, cond: dict, target, t):
        cond_tensor = cond_to_tensor(cond)
        ts, vals = cond_dict_to_arrays(cond)
        cond_packed = (ts, vals)

        key, sub = random.split(key)
        noise = random.normal(sub, x_start.shape, dtype=x_start.dtype)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if cond_packed is not None:
            x_noisy = self._apply_cond_packed(x_noisy, cond_packed)

        pred = self.generator(x_noisy, cond_tensor, t)
        loss, info = self.loss_fn(pred, target)
        return loss, info

    def loss(self, key, x, cond, rewards=None, latent=None, target=None):
        if target is None:
            raise ValueError("ValueDiffusionJax.loss requires `target`.")

        B = int(x.shape[0])
        key, sub = random.split(key)
        t = random.randint(sub, (B,), minval=0, maxval=int(self.n_timesteps), dtype=jnp.int32)
        return self.p_losses(key, x, cond, target, t)
        
    def __call__(self, x: jnp.ndarray, cond_tensor: jnp.ndarray, t: Any):
        """
        Value/scorer forward.
        Args:
        x:          [B, H, transition_dim]
        cond_tensor:[B, C]
        t:          scalar / [B] (python int, scalar array, or [B])
        Returns:
        pred:       [B]
        """

        out = self.generator(x, cond_tensor, t)
        return out
