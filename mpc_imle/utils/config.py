import os
import collections
import importlib
import pickle
from dotenv import load_dotenv, dotenv_values

load_dotenv()
DEVICE = os.getenv("DEVICE")


def import_class(_class):
    if type(_class) is not str:
        return _class
    repo_name = __name__.split('.')[0]
    module_name = '.'.join(_class.split('.')[:-1])
    class_name = _class.split('.')[-1]
    module = importlib.import_module(f'{repo_name}.{module_name}')
    _class = getattr(module, class_name)
    print(f'[ utils/config ] Imported {repo_name}.{module_name}:{class_name}')
    return _class


def get_default_device():
    """Get default JAX device (imported only when called)"""
    import jax
    devices = jax.devices()
    if devices:
        return devices[0]
    else:
        return jax.devices('cpu')[0]


class Config(collections.abc.Mapping):

    def __init__(self, _class, verbose=True, savepath=None, device=None, **kwargs):
        self._class = import_class(_class)
        self._device = device if device else get_default_device()
        self._dict = {}

        for key, val in kwargs.items():
            self._dict[key] = val

        if verbose:
            print(self)

        if savepath is not None:
            savepath = os.path.join(*savepath) if type(savepath) is tuple else savepath
            self._device_str = str(self._device)
            pickle.dump(self, open(savepath, 'wb'))
            print(f'[ utils/config ] Saved config to: {savepath}\n')

    def __repr__(self):
        string = f'\n[ utils/config ] Config: {self._class}\n'
        for key in sorted(self._dict.keys()):
            val = self._dict[key]
            string += f'    {key}: {val}\n'
        return string

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, item):
        return self._dict[item]

    def __len__(self):
        return len(self._dict)

    def __getattr__(self, attr):
        if attr == '_dict' and '_dict' not in vars(self):
            self._dict = {}
            return self._dict
        try:
            return self._dict[attr]
        except KeyError:
            raise AttributeError(attr)

    def __call__(self, *args, **kwargs):
        """
        Instantiate the class with config parameters.
        """
        merged_kwargs = {**self._dict, **kwargs}
        
        if args:
            instance = self._class(*args, **merged_kwargs)
        else:
            instance = self._class(**merged_kwargs)
        
        print(f"[ utils/config ] Instantiated {type(instance).__name__}")
        
        return instance

    def __getstate__(self):
        """Handle pickling by converting Device to string"""
        state = self.__dict__.copy()
        if hasattr(self, '_device'):
            state['_device_str'] = str(self._device)
            del state['_device']
        return state

    def __setstate__(self, state):
        """Handle unpickling by reconstructing Device from string"""
        if '_device_str' in state:
            device_str = state.pop('_device_str')
            self._device = device_str
        self.__dict__.update(state)