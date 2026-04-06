import torch
import torch.nn as nn
from collections import namedtuple
import os
from dotenv import load_dotenv

from .helpers.losses import (
    Losses,
)

from .helpers.sampling import (
    sort_by_values,
    cond_to_tensor
)

from mpc_imle.utils.timer import Timer_Better
load_dotenv() 

DEVICE = os.getenv("DEVICE")

Sample = namedtuple("Sample", "trajectories values")

class IMLEModel(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        observation_dim,
        action_dim,
        loss_type="l2",
        sample_factor=10,
        noise_coef=0.1,
        staleness=20,
        latent_dim=8,
        action_weight=1.0,
        loss_discount=1.0,
        alpha=0.0,
        offset=0.0,
        sigma=1.0,
        loss_weights=None,
        use_inv_model=False, # Deprecated
        inv_hidden_dim=256, # Deprecated
        chunk_size = 1024, # Deprecated
        eps_radius = 0.1,
        rs = False,
        eps_threshold = 0.8,
    ):
        super().__init__()
        # Env Properties
        self.horizon = horizon
        self.generator = model
        self.transition_dim = observation_dim + action_dim
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        # IMLE Properties
        self.sample_factor = sample_factor
        self.noise_coef = noise_coef
        self.staleness = staleness
        self.latent_dim = latent_dim
        self.eps_radius = eps_radius
        self.rs = rs
        self.eps_threshold = eps_threshold
       
        # Loss Properties
        self.alpha = alpha
        self.offset = offset
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights, self.transition_dim)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)
        self.val_loss_fn = None

        # Sampling
        self.sigma = sigma

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **kwargs):
        """
        Forward pass through the generator.

        Returns:
            torch.Tensor: Generated trajectories [batch_size, horizon, output_dim].
        """
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, self.latent_dim, horizon)

        cond_tensor = cond_to_tensor(cond)
        x = self.sigma * torch.randn(shape, device=cond_tensor.device)
        t_gen = Timer_Better(cond_tensor.device, name="generator")
        trajectories = self.generator(x, cond_tensor)
        t_gen()
        values = torch.zeros(batch_size, device=trajectories.device)

        if kwargs.get('guide', False):
            sample_fn = kwargs.get('sample_fn', None)
            t_guid = Timer_Better(trajectories.device, name="guidance")
            _, values = sample_fn(self, trajectories, cond, guide = kwargs['guide'])
            t_guid()
            t_rank = Timer_Better(trajectories.device, name="ranking")
            trajectories, values = sort_by_values(trajectories, values)
            t_rank()

        return Sample(trajectories=trajectories, values=values)

    def get_loss_weights(self, action_weight, discount, weights_dict, dims):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(dims, dtype=torch.float32)
        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    def forward(self, cond, *args, **kwargs):
        """
        Forward pass through the generator.

        Returns:
            torch.Tensor: Generated trajectories [batch_size, horizon, output_dim].
        """
        return self.conditional_sample(cond=cond, *args, **kwargs)
    
    def torch_nn(self, x, generated, zs):
        # Standard nearest neighbor approach
        K = self.sample_factor
        batch_size, _, _ = x.shape
        # Reshape generated samples to group per input x_i
        generated = generated.view(batch_size, K, *x.shape[1:])
        
        # Compute nearest neighbors individually per batch item
        dists = torch.norm(generated - x.unsqueeze(1), dim=(2, 3))

        if (self.rs):
            reject_mask = dists < self.eps_radius  # [B, K]
            dists = dists.masked_fill(reject_mask, float('inf'))  # Mask out rejected ones

        nn_indices = dists.argmin(dim=1)
        nns = torch.arange(batch_size, device=x.device) * K + nn_indices
        
        # Add noise to selected latents
        noise = torch.randn_like(zs[nns]) * self.noise_coef
        imle_nn_z = zs[nns] + noise
        
        return imle_nn_z # [B, latent_dim, H]
    
    def adaptive_generate_latent(self, x, cond, rewards, latents, tau_i_prev):
        """
        Adaptive IMLE for conditional generation with per-sample resampling.

        Args:
            x       : [B, H, D] target trajectories
            cond    : conditioning dict
            rewards : (unused)
            latents : [B, latent_dim, H] current latent codes
            tau_i_prev : [B] previous distance thresholds per sample

        Returns:
            updated_latents : [B, latent_dim, H]
            tau_i_new       : [B] new tau values after resampling
        """
        batch_size, horizon, dim = x.shape
        K = self.sample_factor

        cond_tensor = cond_to_tensor(cond)

        self.generator.eval()
        with torch.no_grad():
            current_gen = self.generator(latents, cond_tensor)  # [B, H, D]
            dist = torch.norm(current_gen - x, dim=(1, 2))       # [B]

        update_mask = dist <= tau_i_prev

        updated_latents = latents.clone()
        dist_new = dist.clone()

        if update_mask.any():
            x_upd = x[update_mask]
            cond_upd_tensor = cond_tensor[update_mask]

            with torch.no_grad():
                # Resample new candidates
                zs = torch.randn(len(x_upd) * K, self.latent_dim, horizon, device=x.device)
                cond_rep = cond_upd_tensor.repeat_interleave(K, dim=0)
                generated = self.generator(zs, cond_rep)  # [B*K, H, D]
                generated = generated.view(len(x_upd), K, horizon, dim)

                dists = torch.norm(generated - x_upd.unsqueeze(1), dim=(2, 3))  # [B, K]
                nn_indices = dists.argmin(dim=1)

                selected_zs = zs.view(len(x_upd), K, self.latent_dim, horizon)[
                    torch.arange(len(x_upd)), nn_indices
                ]
                noise = torch.randn_like(selected_zs) * self.noise_coef
                selected_zs = selected_zs + noise

                updated_gen = self.generator(selected_zs, cond_upd_tensor)
                new_dist = torch.norm(updated_gen - x_upd, dim=(1, 2))

            updated_latents[update_mask] = selected_zs
            dist_new[update_mask] = new_dist

        tau_i_new = self.eps_threshold * dist_new

        self.generator.train()

        return updated_latents, tau_i_new

    def generate_latent(self, x, cond, rewards):
        """
        Combined implementation of latent generation with option to use DCI-KNN or standard nearest neighbor.
        
        Args:
            x:        [B, H, D] - Target trajectories
            cond:     Dict[str, Tensor] - Condition inputs
            rewards:  Ignored here, kept for API consistency
        
        Returns:
            latent: [B, latent_dim, H] - Perturbed latent code
        """
        batch_size, horizon, dim = x.shape
        K = self.sample_factor
        
        # Process input if using inverse model
        if self.use_inv_model:
            x = x[:, :, self.action_dim:]
            dim = self.observation_dim
        
        # Generate candidate latent codes and expand conditioning
        zs = torch.randn(batch_size * K, self.latent_dim, horizon, device=x.device)
        cond_tensor = cond_to_tensor(cond).detach()
        cond_imle = cond_tensor.repeat_interleave(K, dim=0)

        # Generate trajectories
        self.generator.eval()
        with torch.no_grad():
            generated = self.generator(zs, cond_imle)  # [B*K, H, D]
        
        imle_nn_z = self.torch_nn(x, generated, zs)

        self.generator.train()

        return imle_nn_z
        
    def loss(self, x, cond, rewards, latent):
        cond_tensor = cond_to_tensor(cond)
        self.generator.train()
        outs = self.generator(latent, cond_tensor)
        
        # Interpolates between [0,1] when alpha=0 and [1,1] when alpha=1)
        reward_weights = self.offset + self.alpha + (1 - self.alpha) * rewards

        loss = self.loss_fn(outs, x, reward_weights)

        return loss