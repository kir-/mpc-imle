import os

from mpc_imle.utils import watch
from dotenv import load_dotenv

load_dotenv() 

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('staleness', 'S'),
    ('sample_factor', 'SF'),
    ('reward_weighting', 'RW'),
    ## value kwargs
    ('discount', 'd'),
    ('jax_flag', 'j'),
]

logbase = 'logs'

base = {
    'imle': {
        ## model
        'model': 'models.TemporalUnetIMLE',
        'imle': 'models.IMLEModel',
        'horizon': 32,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'dim_mults': (1, 4, 16),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',
        'staleness': 1,  # Number of epochs before updating nearest neighbors
        'sample_factor': 4,
        'noise_coef': 0.0001,
        'z_dim': 32, # Feature Expansion Factor
        'latent_dim': 16, # Latent Space Dimension
        'cond_factor': 1, # Number of Conditioning vectors
        'alpha': 0, # Reward interpolation
        'offset': 0, # Reward Offset
        'awr': False,  # Deprecated
        'exp_min': 0.1,
        'exp_max': 3.0,
        'beta': 1.0,
        'use_inv_model': False, # Deprecated
        'inv_hidden_dim': 128,  # Deprecated
        'rs': False,
        'eps_radius': 0.1,
        'eps_threshold': 0.98,
        'jax_flag': 0,

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'reward_weighting': False,
        'max_path_length': 1000,
        'max_samples': None,

        ## serialization
        'logbase': logbase,
        'prefix': 'imle/defaults',
        'exp_name': watch(args_to_watch),
        'suffix': '',

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 100000,
        'batch_size': 32,
        'learning_rate': 0.001,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 10000,
        'sample_freq': 20000,
        'val_freq': 10000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'n_samples': 10,
        'bucket': None,
        'device': os.getenv("DEVICE"),
        'seed': 42,
        'use_layer_norm': False,
        'warmup_steps': 10000,
        'use_adamw': False,
        'ada_imle': False,
        'sampling_type': "uniform",
    },

    'values': {
        'model': 'models.ValueFunction',
        'horizon': 32,
        'dim_mults': (1, 2, 4, 8),
        'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'val_freq': 10000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': os.getenv("DEVICE"),
        'seed': 42,
    },

    'plan': {
        'policy': 'sampling.GuidedPolicy',
        'guide': 'sampling.IMLEValueGuide',
        'max_episode_length': 1000,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': os.getenv("DEVICE"),
        'seed': None,
        'jax': False,
        'jax_guide': False,
        'jax_flag': 0,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## IMLE model
        'horizon': 32,
        'n_diffusion_steps': 20,
        'staleness': 1,  # Sync with model staleness
        'sample_factor': 4,
        'reward_weighting': False,
        'sigma': 1,

        ## value function
        'discount': 0.99,

        ## loading
        'diffusion_loadpath': 'f:imle/defaults_H{horizon}_S{staleness}_SF{sample_factor}_RW{reward_weighting}_j{jax_flag}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}_j{jax_flag}',

        'value_epoch': 'latest',
        'diffusion_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },
}


#------------------------ overrides ------------------------#


hopper_medium_expert_v2 = {
    'plan': {
        'scale': 0.0001,
        't_stopgrad': 4,
    },
}


halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = {
    'imle': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
        'attention': True,
    },
    'values': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
    },
    'plan': {
        'horizon': 4,
        'scale': 0.001,
        't_stopgrad': 4,
    },
}
