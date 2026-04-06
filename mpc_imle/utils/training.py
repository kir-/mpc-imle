import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import einops
import pdb
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv() 

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .lr_scheduler import get_linear_schedule_with_warmup

import wandb

use_wandb = os.getenv("WANDB", "FALSE").upper() == "TRUE"

DEVICE = os.getenv("DEVICE_SPECIFIC")

if (os.getenv("APPLE_DEVICE", "FALSE").upper() == "TRUE"):
    import torch.multiprocessing as mp
    mp.set_start_method('fork', force=True)


if (os.getenv("CUDNN_BACKEND", "FALSE").upper() == "TRUE"):
    torch.backends.cudnn.benchmark = True 

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        val_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        val_size=100,
        wandb_run=None,
        use_imle=False,
        horizon=10,
        staleness=1,
        latent_dim=6,
        bucket=None,
        warmup_steps=0,
        use_adamw=False,
        total_training_steps=100000,
        ada_imle=False,
        value_function=False
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.val_freq = val_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel
        self.value_function = value_function

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset_size = len(dataset)
        val_size = min(val_size, self.dataset_size)
        train_size = self.dataset_size - val_size
        indices = np.random.permutation(self.dataset_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size] 
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_dataset_size = len(self.train_dataset)
        self.train_remainder = self.train_dataset_size % self.batch_size 
        self.num_batches_per_epoch = self.train_dataset_size // self.batch_size 

        # IMLE specific variables
        self.use_imle = use_imle
        self.ada_imle = ada_imle
        if use_imle:
            self.latent_dim = latent_dim
            self.staleness = staleness
            self.horizon = horizon

        self.dataloader = cycle(DataLoader(
            self.train_dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.val_dataloader = cycle(DataLoader(
            self.val_dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(DataLoader(
            self.train_dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer

        if use_adamw:
            self.optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=train_lr)
        else:
            self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        if (warmup_steps == 0):
            self.use_warmup = False
        else:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_training_steps)
            self.use_warmup = True

        self.logdir = results_folder
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0
        self.wandb_run = wandb_run

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, max_train_steps, max_epochs):
        timer = Timer()
        if (self.use_imle and self.ada_imle):
            latent_total = torch.randn((self.dataset_size, self.latent_dim, self.horizon), device='cpu')
            tau_total = torch.full((self.dataset_size,), float('inf'), device='cpu')
        elif (self.use_imle and self.staleness > 1):
            latent_total = torch.empty((self.dataset_size, self.latent_dim, self.horizon), device='cpu')
        
        for epoch in range(max_epochs):
            print(f'Epoch {epoch+1} / {max_epochs} | {self.logdir}')
            
            for _ in tqdm(range(self.num_batches_per_epoch), desc=f"Epoch {epoch+1}", unit="batch"):
                step_loss = 0
                for _ in range(self.gradient_accumulate_every):
                    idx, batch = next(self.dataloader)
                    batch = batch_to_device(batch)
                    if self.use_imle:
                        if (self.ada_imle):
                            latent, tau_i = self.model.adaptive_generate_latent(*batch, latent_total[idx].to(DEVICE), tau_total[idx].to(DEVICE))
                            latent_total[idx] = latent.cpu()
                            tau_total[idx] = tau_i.cpu()
                        elif (self.staleness > 1):
                            if (epoch % self.staleness == 0):
                                latent = self.model.generate_latent(*batch)
                                latent_total[idx] = latent.cpu()
                            else:
                                latent = latent_total[idx].to(DEVICE)
                        else:
                            latent = self.model.generate_latent(*batch)
                        loss, infos = self.model.loss(*batch, latent=latent)
                    else:
                        loss, infos = self.model.loss(*batch)
                    loss = loss / self.gradient_accumulate_every
                    step_loss += loss.item()
                    loss.backward()

                self.optimizer.step()
                if (self.use_warmup == True):
                    self.scheduler.step()
                self.optimizer.zero_grad()

                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                if (self.step % self.save_freq == 0):
                    label = self.step
                    self.save(label)

                if self.step % self.log_freq == 0:
                    infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                    print(f'{self.step}: {step_loss:8.4f} | {infos_str} | t: {timer():8.4f}')

                if self.wandb_run:
                    if (self.step % self.val_freq == 0):
                        val_loss = self.compute_validation_loss(epoch)
                        self.wandb_run.log({"validation_loss": val_loss, "train_loss": step_loss, "step": self.step})
                    elif (self.step % self.log_freq == 0):
                        self.wandb_run.log({"train_loss": step_loss, "step": self.step})
                
                # if self.step == 0 and self.sample_freq:
                #     self.render_reference(self.n_reference)

                # if self.sample_freq and self.step % self.sample_freq == 0:
                #     self.render_samples(n_samples=self.n_samples)

                self.step += 1
                if (self.step == max_train_steps):
                    print("Maximum Training Steps Hit: ", max_train_steps)
                    return

        
    def compute_validation_loss(self, epoch):
        self.model.eval()
        total_val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for _ in range(len(self.val_dataset)):
                _, batch = next(self.val_dataloader)
                batch = batch_to_device(batch)
                x, cond, rewards = batch
                batch_size = x.shape[0]
                
                t = torch.zeros(batch_size, device=x.device) # time
                weights = torch.ones(batch_size, device=x.device)  # uniform weights

                if self.value_function:
                    sample = self.model(x, cond, t)
                    target = rewards
                else:
                    sample = self.model.conditional_sample(cond).trajectories
                    target = x
                
                if self.model.val_loss_fn:
                    loss, _ = self.model.val_loss_fn(sample, target, weights)
                else:
                    loss, _ = self.model.loss_fn(sample, target, weights)

                total_val_loss += loss.item()
                num_batches += 1

        self.model.train()
        return total_val_loss / num_batches 

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{int(epoch)}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{int(epoch)}.pt')
        data = torch.load(loadpath, map_location=torch.device(DEVICE), weights_only=True)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, DEVICE)

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)
