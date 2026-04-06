from collections import namedtuple
import numpy as np
import jax.numpy as jnp

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer


Batch = namedtuple('Batch', 'trajectories conditions values')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


class SequenceDatasetJax:
    """JAX version of sequence dataset for trajectory data."""

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None, 
        discount=0.99, normed=False, reward_weighting=False):
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
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        
        if reward_weighting:
            # Compute all discounted returns
            self.raw_values = []
            for i in range(len(self.indices)):
                path_ind, start, _ = self.indices[i]
                rewards = self.fields['rewards'][path_ind, start:]
                discounts = self.discounts[:len(rewards)]
                value = (discounts * rewards).sum()
                self.raw_values.append(value)
            self.raw_values = np.array(self.raw_values)
            self.vmin = self.raw_values.min()
            self.vmax = self.raw_values.max()
        else:
            self.raw_values = np.ones(len(self.indices), dtype=np.float32)
            self.vmin = 0.0
            self.vmax = 1.0

        print(fields)

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = np.asarray(self.fields[key]).reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(
                self.n_episodes, self.max_path_length, -1
            ).astype(np.float32)

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
    
    def normalize_value(self, value):
        '''
            normalizes rewards between 0 and 1
        '''
        normed = (value - self.vmin) / (self.vmax - self.vmin + 1e-8)
        assert np.all(normed >= 0 - 1e-5) and np.all(normed <= 1.0 + 1e-5), \
            f"Normalized values out of expected [0, 1] range: [{normed.min()}, {normed.max()}]"
        return normed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = np.asarray(self.fields.normed_observations[path_ind, start:end])
        actions = np.asarray(self.fields.normed_actions[path_ind, start:end])

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        value = self.raw_values[idx]
        value = self.normalize_value(float(value))

        value = np.array([value], dtype=np.float32)
        batch = Batch(
            trajectories.astype(np.float32),
            {k: np.asarray(v, dtype=np.float32) for k, v in conditions.items()},
            value,
        )
        return idx, batch

    def __iter__(self):
        """Iterate through dataset in order."""
        for i in range(len(self)):
            yield self[i]

    def sample_batch(self, batch_size, rng=None):
        idxs = np.random.randint(0, len(self.indices), size=(batch_size,))

        path = self.indices[idxs, 0].astype(np.int32)   # (B,)
        start = self.indices[idxs, 1].astype(np.int32)  # (B,)

        t = np.arange(self.horizon, dtype=np.int32)[None, :]  # (1,H)
        time_idx = start[:, None] + t                          # (B,H)

        obs = self.fields['normed_observations'][path[:, None], time_idx]  # (B,H,obs)
        act = self.fields['normed_actions'][path[:, None], time_idx]       # (B,H,act)

        traj = np.concatenate([act, obs], axis=-1).astype(np.float32)       # (B,H,D)
        cond0 = obs[:, 0, :].astype(np.float32)                             # (B,obs)

        # default "values" in base dataset = raw_values normalized [0,1]
        vals = self.raw_values[idxs].astype(np.float32)                     # (B,)
        vals = (vals - self.vmin) / (self.vmax - self.vmin + 1e-8)
        vals = vals[:, None].astype(np.float32)                             # (B,1)

        # single host->device transfer
        return Batch(jnp.asarray(traj), {0: jnp.asarray(cond0)}, jnp.asarray(vals))


class GoalDatasetJax(SequenceDatasetJax):
    """JAX dataset with goal conditioning."""

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDatasetJax(SequenceDatasetJax):
    '''
        JAX dataset with value function labels
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
            value = self[i][1].values.item()
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
        _, batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = np.asarray(self.fields['rewards'][path_ind, start:])
        discounts = np.asarray(self.discounts[:len(rewards)])
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(float(value))
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(batch.trajectories, batch.conditions, value)
        return idx, value_batch
