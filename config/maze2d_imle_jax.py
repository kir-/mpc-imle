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
    ('jax_flag', 'j'),
    ## value kwargs
    ('discount', 'd'),
]

plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ('jax_flag', 'j'),
    ##
    ('conditional', 'cond'),
]

logbase = 'logs'

base = {
    'imle': {
        ## model
        'model': 'models.TemporalUnetIMLEJax',
        'imle': 'models.IMLEModelJax',
        'horizon': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'dim_mults': (1, 2, 4, 8, 16),
        'attention': False,
        'renderer': 'utils.Maze2dRenderer',
        'staleness': 1,  # Number of epochs before updating nearest neighbors
        'sample_factor': 20,
        'noise_coef': 0.01,  # Noise coefficient for perturbing z-space,
        'z_dim': 32, # Feature Expansion Factor
        'latent_dim': 3, # Latent Space Dimension
        'cond_factor': 2, # Number of Conditioning vectors
        'alpha': 0, # Reward interpolation
        'offset': 0, # Reward Offset
        'use_inv_model': False,
        'inv_hidden_dim': 128,
        'jax_flag': 1,


        ## dataset
        'loader': 'datasets.GoalDatasetJax',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'reward_weighting': False,
        'max_path_length': 40000,

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
        'learning_rate': 0.0001,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 5000,
        'val_freq': 10000,
        'sample_freq': 1000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': os.getenv("DEVICE"),
        'use_layer_norm': True,
        'warmup_steps': 0,
        'use_adamw': False,
    },

    'plan': {
        'policy': 'sampling.ModelPolicyJax',
        'batch_size': 1,
        'device': os.getenv("DEVICE"),
        'jax_flag': 1,
        'seed': None,

        ## IMLE
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',
        'staleness': 1,  # Sync with model staleness
        'sample_factor': 20,
        'reward_weighting': False,

        ## serialization
        'loadbase': None,
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/imle',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'f:imle/defaults_H{horizon}_S{staleness}_SF{sample_factor}_RW{reward_weighting}_j{jax_flag}',
        'diffusion_epoch': 'latest',

    },
}


#------------------------ overrides ------------------------#


'''
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
'''

maze2d_umaze_v1 = {
    'imle': {
        'horizon': 128,
    },
    'plan': {
        'horizon': 128,
    },
}

maze2d_large_v1 = {
    'imle': {
        'horizon': 384,
    },
    'plan': {
        'horizon': 384,
    },
}
