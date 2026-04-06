from collections import namedtuple
import numpy as np
import torch
from enum import Enum

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer


Batch = namedtuple('Batch', 'trajectories conditions values')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SamplingType(str, Enum):
    UNIFORM = "uniform"
    REWARD_WEIGHTED = "reward_weighted"
    STRATIFIED = "stratified"

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000, max_samples=None,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None, discount=0.99, normed=False, reward_weighting=False, awr=False, exp_min=0.1, exp_max=2, beta=1, sampling_type="uniform"):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        
        all_indices = self.make_indices(fields.path_lengths, horizon)
        self.indices = all_indices

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.sampling_type = SamplingType(sampling_type)

        # Exp reweighting params
        self.beta = beta
        self.exp_min = exp_min
        self.exp_max = exp_max

        raw_values = []
        for i in range(len(all_indices)):
            path_ind, start, _ = all_indices[i]
            rewards = fields['rewards'][path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            value = (discounts * rewards).sum()
            raw_values.append(value)

        raw_values = np.array(raw_values)  # shape [len(all_indices)]
        if max_samples is not None:
            np.random.seed(seed)
            if self.sampling_type == SamplingType.STRATIFIED:
                sampled_indices = self.stratified_sampling_probs(raw_values, max_samples, len(all_indices))
            elif self.sampling_type == SamplingType.REWARD_WEIGHTED:
                self._init_reward_weighting_params(raw_values)
                pi = self.exp_sampling_probs(raw_values)
                sampled_indices = np.random.choice(len(raw_values), size=max_samples, replace=True, p=pi)
            else:
                sampled_indices = np.random.choice(len(all_indices), size=max_samples, replace=True)
            
            self.indices = all_indices[sampled_indices]
            self.raw_values = raw_values[sampled_indices]
        else:
            self.raw_values = raw_values

        if reward_weighting:
            self._init_reward_weighting_params(self.raw_values)
        else:
            self.raw_values = np.ones(len(self.indices), dtype=np.float32)
            self.baseline = 0.0
            self.vmin = 0.0
            self.vmax = 1.0

        print(fields)

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}
    
    def _init_reward_weighting_params(self, values):
        """Initialize parameters needed for reward weighting and exponential normalization"""
        self.vmin = values.min()
        self.vmax = values.max()
        self.baseline = np.median(values)
        self.std_dev = np.median(np.abs(values - np.median(values))) + 1e-8  # avoid divide-by-zero
    
    def normalize_value(self, value):
        '''
            normalizes rewards between 0 and 1
        '''
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin + 1e-8)
        assert np.all(normed >= 0 - 1e-5) and np.all(normed <= 1.0 + 1e-5), f"Normalized values out of expected [0, 1] range: [{normed.min()}, {normed.max()}]"
        
        return normed
    
    def exp_normalize_value(self, value):
        weights = np.exp(((value) - self.baseline) / (self.std_dev * self.beta))
        weights = np.clip(weights, self.exp_min, self.exp_max)
        return weights
    
    def exp_sampling_probs(self, values):
        weights = self.exp_normalize_value(values)
        return weights / weights.sum()
    
    def stratified_sampling_probs(self, values, max_samples, N, K = 10):
        sorted_indices = np.argsort(values)
        bin_size = N // K
        remainder = N % K
        
        bin_assignments = np.zeros(N, dtype=int)
        start_idx = 0
        
        for k in range(K):
            current_bin_size = bin_size + (1 if k < remainder else 0)
            end_idx = start_idx + current_bin_size
            bin_assignments[sorted_indices[start_idx:end_idx]] = k
            start_idx = end_idx
        
        samples_per_bin = max_samples // K
        extra_samples = max_samples % K
        sampled_indices = []
        
        for k in range(K):
            bin_indices = np.where(bin_assignments == k)[0] 
            n_samples_k = samples_per_bin + (1 if k < extra_samples else 0)
            
            sampled_k = np.random.choice(bin_indices, size=n_samples_k, replace=True)
            sampled_indices.extend(sampled_k)

        return np.array(sampled_indices)
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        value = self.raw_values[idx]
        if (self.reward_weighting):
            value = self.exp_normalize_value(value)
        else:
            value = self.normalize_value(value)

        value = np.array([value], dtype=np.float32)
        batch = Batch(trajectories, conditions, value)
        return idx, batch


class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        idx, batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        values = (discounts * rewards).sum()
        if self.normed:
            values = self.normalize_value(values)
        values = np.array([values], dtype=np.float32)
        value_batch = ValueBatch(batch.trajectories, batch.conditions, values)
        return idx, value_batch
