import jax
import jax.numpy as jnp
from flax import linen as nn

#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#


class WeightedLoss(nn.Module):
    weights: jnp.ndarray
    action_dim: int

    @nn.compact
    def __call__(self, pred, targ, reward_weights=None):
        """
        pred, targ : [batch_size, horizon, transition_dim]
        weights    : [horizon, transition_dim]
        reward_weights : [batch_size] or None
        """
        loss = self._loss(pred, targ)
        # Apply per-element weights: [H, D] -> broadcasted to [B, H, D]
        weighted = loss * self.weights  # [B, H, D]

        # Mean over time and dimension per sample: [B]
        per_traj_loss = weighted.mean(axis=(1, 2))  # [B]

        # If reward_weights is None, use identity weights (ones)
        if reward_weights is None:
            reward_weights = jnp.ones_like(per_traj_loss)
        else:
            reward_weights = reward_weights.reshape(-1)  # ensure [B]
        per_traj_loss = per_traj_loss * reward_weights  # [B]

        # Final loss: mean over batch
        final_loss = per_traj_loss.mean()

        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()

        info = {'a0_loss': a0_loss}

        return final_loss, info

    def _loss(self, pred, targ):
        """To be overridden by subclasses"""
        raise NotImplementedError


class ValueLoss(nn.Module):
    @nn.compact
    def __call__(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        # Compute correlation using JAX (JIT-compatible)
        corr = self._compute_correlation(pred, targ)

        info = {
            'mean_pred': pred.mean(),
            'mean_targ': targ.mean(),
            'min_pred': pred.min(),
            'min_targ': targ.min(),
            'max_pred': pred.max(),
            'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

    def _loss(self, pred, targ):
        """To be overridden by subclasses"""
        raise NotImplementedError

    @staticmethod
    def _compute_correlation(pred, targ):
        """
        Compute Pearson correlation using JAX operations.
        Works inside JIT (no NumPy or Python control flow).
        """
        pred_flat = pred.reshape(-1)
        targ_flat = targ.reshape(-1)
        
        # Standardize
        pred_mean = pred_flat.mean()
        targ_mean = targ_flat.mean()
        pred_std = pred_flat.std() + 1e-8
        targ_std = targ_flat.std() + 1e-8
        
        pred_norm = (pred_flat - pred_mean) / pred_std
        targ_norm = (targ_flat - targ_mean) / targ_std
        
        # Pearson correlation coefficient
        corr = (pred_norm * targ_norm).mean()
        
        return corr


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return jnp.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return (pred - targ) ** 2


class ValueL1(ValueLoss):
    def _loss(self, pred, targ):
        return jnp.abs(pred - targ)


class ValueL2(ValueLoss):
    def _loss(self, pred, targ):
        return (pred - targ) ** 2


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
}