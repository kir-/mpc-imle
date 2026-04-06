import os
import pickle
import glob
import torch
import pdb
from dotenv import load_dotenv, dotenv_values

load_dotenv() 
from collections import namedtuple

Experiment = namedtuple('GenerativeModel', 'dataset renderer model diffusion ema trainer epoch')
DEVICE = os.getenv("DEVICE_SPECIFIC")

def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(loadpath, jax=False):
    states = glob.glob1(os.path.join(*loadpath), 'state_*')
    latest_epoch = -1
    for state in states:
        if jax:
            epoch = int(state.replace('state_', '').replace('.pkl', ''))
        else:
            epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    print(config)
    return config

def overwrite_config_seed(config_obj, new_seed):
    if hasattr(config_obj, '_class') and hasattr(config_obj, '_dict'):
        new_dict = dict(config_obj._dict)
        new_dict['seed'] = new_seed
        return type(config_obj)(config_obj._class, **new_dict)
    else:
        raise TypeError("Cannot overwrite seed; unexpected config object type.")

def load_experiment(*loadpath, epoch='latest', device=DEVICE, seed=None, generator='diffusion_config.pkl', jax=False):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    if (seed != None):
        dataset_config = overwrite_config_seed(dataset_config, seed)
    render_config = load_config(*loadpath, 'render_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')

    diffusion_config = load_config(*loadpath, generator)
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    dataset = dataset_config()
    renderer = render_config()
    model = model_config()
        
    if jax:
        diffusion = diffusion_config(generator=model)
    else:
        diffusion = diffusion_config(model)
        
    trainer = trainer_config(diffusion, dataset, renderer)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath, jax=jax)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    if jax:
        return Experiment(dataset, renderer, model, diffusion, diffusion, trainer, epoch)
    else:
        return Experiment(dataset, renderer, model, diffusion, trainer.ema_model, trainer, epoch)

def check_compatibility(experiment_1, experiment_2):
    '''
        returns True if `experiment_1 and `experiment_2` have
        the same normalizers and number of diffusion steps
    '''
    normalizers_1 = experiment_1.dataset.normalizer.get_field_normalizers()
    normalizers_2 = experiment_2.dataset.normalizer.get_field_normalizers()
    for key in normalizers_1:
        norm_1 = type(normalizers_1[key])
        norm_2 = type(normalizers_2[key])
        if (norm_1 != norm_2) and not (norm_1.__name__ == norm_2.__name__):
            print(f'[Warning] Normalizer class mismatch for {key}: {norm_1} vs {norm_2}')

    # _modified: commented because imle does not have n_timesteps

    # n_steps_1 = experiment_1.diffusion.n_timesteps
    # n_steps_2 = experiment_2.diffusion.n_timesteps
    # assert n_steps_1 == n_steps_2, \
    #     ('Number of timesteps should match between diffusion experiments, '
    #     f'found {n_steps_1} and {n_steps_2}')
