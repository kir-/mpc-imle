import collections
import numpy as np
import os
import torch
import pdb
from dotenv import load_dotenv, dotenv_values
import jax
import jax.numpy as jnp

load_dotenv() 
DTYPE = torch.float
DEVICE = os.getenv("DEVICE_SPECIFIC")

#-----------------------------------------------------------------------------#
#------------------------------ numpy <--> torch -----------------------------#
#-----------------------------------------------------------------------------#

def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x

def to_torch(x, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEVICE
	if type(x) is dict:
		return {k: to_torch(v, dtype, device) for k, v in x.items()}
	elif torch.is_tensor(x):
		return x.to(device).type(dtype)
	return torch.tensor(x, dtype=dtype, device=device)

def as_tensor(value, device=None): 
     if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], np.ndarray): 
         value = np.array(value) 
     return torch.tensor(value).to(device)

def to_device(x, device=DEVICE):
	if torch.is_tensor(x):
		return x.to(device)
	elif type(x) is dict:
		return {k: to_device(v, device) for k, v in x.items()}
	else:
		raise RuntimeError(f'Unrecognized type in `to_device`: {type(x)}')

def batchify(batch):
	'''
		convert a single dataset item to a batch suitable for passing to a model by
			1) converting np arrays to torch tensors and
			2) and ensuring that everything has a batch dimension
	'''
	fn = lambda x: to_torch(x[None])

	batched_vals = []
	for field in batch._fields:
		val = getattr(batch, field)
		val = apply_dict(fn, val) if type(val) is dict else fn(val)
		batched_vals.append(val)
	return type(batch)(*batched_vals)

def apply_dict(fn, d, *args, **kwargs):
	return {
		k: fn(v, *args, **kwargs)
		for k, v in d.items()
	}

def normalize(x):
	"""
		scales `x` to [0, 1]
	"""
	x = x - x.min()
	x = x / x.max()
	return x

def to_img(x):
    normalized = normalize(x)
    array = to_np(normalized)
    array = np.transpose(array, (1,2,0))
    return (array * 255).astype(np.uint8)

def set_device(device):
	DEVICE = device
	if 'cuda' in device:
		torch.set_default_tensor_type(torch.cuda.FloatTensor)
	else:
		torch.set_default_tensor_type(torch.FloatTensor)

def batch_to_device(batch, device=DEVICE):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)

def _to_str(num):
	if num >= 1e6:
		return f'{(num/1e6):.2f} M'
	else:
		return f'{(num/1e3):.2f} k'

#-----------------------------------------------------------------------------#
#----------------------------- parameter counting ----------------------------#
#-----------------------------------------------------------------------------#

def param_to_module(param):
	module_name = param[::-1].split('.', maxsplit=1)[-1][::-1]
	return module_name

def report_parameters(model, topk=10):
	counts = {k: p.numel() for k, p in model.named_parameters()}
	n_parameters = sum(counts.values())
	print(f'[ utils/arrays ] Total parameters: {_to_str(n_parameters)}')

	modules = dict(model.named_modules())
	sorted_keys = sorted(counts, key=lambda x: -counts[x])
	max_length = max([len(k) for k in sorted_keys])
	for i in range(topk):
		key = sorted_keys[i]
		count = counts[key]
		module = param_to_module(key)
		print(' '*8, f'{key:10}: {_to_str(count)} | {modules[module]}')

	remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
	print(' '*8, f'... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters')
	return n_parameters


def count_parameters(params):
    """Count total parameters in a JAX parameter tree"""
    def _count(x):
        if isinstance(x, jnp.ndarray):
            return x.size
        return 0
    
    return jax.tree_util.tree_reduce(
        lambda a, b: a + _count(b),
        params,
        0
    )


def flatten_params(params, prefix=''):
    """Flatten nested parameter dict into flat dict with full paths"""
    flat = {}
    
    def _flatten(d, path=''):
        if isinstance(d, dict):
            for key, val in d.items():
                new_path = f'{path}/{key}' if path else key
                _flatten(val, new_path)
        elif isinstance(d, (list, tuple)):
            for i, val in enumerate(d):
                new_path = f'{path}[{i}]' if path else f'[{i}]'
                _flatten(val, new_path)
        elif isinstance(d, jnp.ndarray):
            flat[path] = d
    
    _flatten(params, prefix)
    return flat


def report_parameters_jax(params, topk=10):
    """
    Report parameter counts for a JAX model.
    
    Args:
        params: JAX parameter tree (dict of arrays) from model.init()
        topk: Number of top parameters to display
    
    Returns:
        Total parameter count
    """
    # Flatten the parameter tree
    flat_params = flatten_params(params)
    
    # Count parameters per layer
    counts = {k: v.size for k, v in flat_params.items()}
    n_parameters = sum(counts.values())
    print(f'[ utils/arrays ] Total parameters: {_to_str(n_parameters)}')
    
    # Sort by parameter count
    sorted_keys = sorted(counts, key=lambda x: -counts[x])
    max_length = max([len(k) for k in sorted_keys]) if sorted_keys else 0
    
    # Display top-k parameters
    for i in range(min(topk, len(sorted_keys))):
        key = sorted_keys[i]
        count = counts[key]
        print(' ' * 8, f'{key:10}: {_to_str(count)}')
    
    # Display remaining
    if len(sorted_keys) > topk:
        remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
        print(' ' * 8, f'... and {len(counts) - topk} others accounting for {_to_str(remaining_parameters)} parameters')
    
    return n_parameters

def to_jax_array(x):
    """
    Convert common batch elements (torch.Tensor, np.ndarray, dict-of-tensors)
    into a JAX array.
    """
    # Dict case: pick a tensor or stack values depending on your cond format
    if isinstance(x, dict):
        # Minimal behavior (matches what you were doing in init_training_state):
        # use the first value in the dict.
        first_val = list(x.values())[0]
        return jnp.array(to_np(first_val))

    # Torch tensor -> numpy -> jax
    if torch.is_tensor(x):
        return jnp.array(to_np(x))

    # Numpy array or anything that np.array can handle
    return jnp.array(np.array(x))

def batchify_jax(batch):
	'''
		convert a single dataset item to a batch suitable for JAX by
			1) converting np arrays to jax arrays and
			2) ensuring that everything has a batch dimension
	'''
	import jax.numpy as jnp
	
	fn = lambda x: jnp.asarray(x[None])

	batched_vals = []
	for field in batch._fields:
		val = getattr(batch, field)
		val = apply_dict(fn, val) if type(val) is dict else fn(val)
		batched_vals.append(val)
	return type(batch)(*batched_vals)

def concatenate_cond_jax(cond):
    """Concatenate condition dict values, handling 1-key and multi-key cases safely."""
    if isinstance(cond, dict):
        vals = list(cond.values())

        if len(vals) == 1:
            return jnp.asarray(vals[0])

        # Multi-key: concat along feature axis
        return jnp.concatenate([jnp.asarray(v) for v in vals], axis=-1)

    # Not a dict
    return jnp.asarray(cond)


@jax.jit
def nearest_neighbor_jax(generated_latent, x):
	dists = jnp.linalg.norm(generated_latent -x, axis=(2,3))
	nn_indices = jnp.argmin(dists, axis=1)

