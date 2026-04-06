import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mpc_imle.utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):
    def __init__(self, weights, action_dim, persistent=True):
        super().__init__()
        self.register_buffer('weights', weights, persistent=persistent)
        self.action_dim = action_dim

    def forward(self, pred, targ, reward_weights=None):
        """
        pred, targ : [batch_size, horizon, transition_dim]
        weights    : [batch_size, 1]
        """

        loss = self._loss(pred, targ)
        # Apply per-element weights: [H, D] -> broadcasted to [B, H, D]
        weighted = loss * self.weights  # [B, H, D]

        # Mean over time and dimension per sample: [B]
        per_traj_loss = weighted.mean(dim=[1, 2])  # [B]

        # If reward_weights is None, use identity weights (ones)
        if reward_weights is None:
            reward_weights = torch.ones_like(per_traj_loss)
        else:
            reward_weights = reward_weights.view(-1)  # ensure [B]
        per_traj_loss = per_traj_loss * reward_weights  # [B]

        # Final loss: mean over batch
        final_loss = per_traj_loss.mean()

        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()

        return final_loss, {'a0_loss': a0_loss}
    
class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, pred, targ, weight=None):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(),
                utils.to_np(targ).squeeze()
            )[0,1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
}
