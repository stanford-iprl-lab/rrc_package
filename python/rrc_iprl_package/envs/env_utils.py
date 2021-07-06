import inspect
import numpy as np

from gym.spaces import Box
from gym.spaces import Discrete
from gym.spaces import MultiDiscrete
from gym.spaces import MultiBinary
from gym.spaces import Tuple
from gym.spaces import Dict
from gym.spaces import utils


def configurable(pickleable: bool = False):
    """Class decorator to allow injection of constructor arguments.

    Example usage:
    >>> @configurable()
    ... class A:
    ...     def __init__(self, b=None, c=2, d='Wow'):
    ...         ...

    >>> set_env_params(A, {'b': 10, 'c': 20})
    >>> a = A()      # b=10, c=20, d='Wow'
    >>> a = A(b=30)  # b=30, c=20, d='Wow'

    Args:
        pickleable: Whether this class is pickleable. If true, causes the pickle
            state to include the constructor arguments.
    """
    # pylint: disable=protected-access,invalid-name

    def cls_decorator(cls):
        assert inspect.isclass(cls)

        # Overwrite the class constructor to pass arguments from the config.
        base_init = cls.__init__

        def __init__(self, *args, **kwargs):

            if pickleable:
                self._pkl_env_args = args
                self._pkl_env_kwargs = kwargs

            base_init(self, *args, **kwargs)

        cls.__init__ = __init__

        # If the class is pickleable, overwrite the state methods to save
        # the constructor arguments
        if pickleable:
            # Use same pickle keys as gym.utils.ezpickle for backwards compat.
            PKL_ARGS_KEY = '_ezpickle_args'
            PKL_KWARGS_KEY = '_ezpickle_kwargs'
            def __getstate__(self):
                return {
                   PKL_ARGS_KEY: self._pkl_env_args,
                    PKL_KWARGS_KEY: self._pkl_env_kwargs,
                }
            cls.__getstate__ = __getstate__

            def __setstate__(self, data):
                saved_args = data[PKL_ARGS_KEY]
                saved_kwargs = data[PKL_KWARGS_KEY]

                inst = type(self)(*saved_args, **saved_kwargs)
                self.__dict__.update(inst.__dict__)

            cls.__setstate__ = __setstate__

        return cls

    # pylint: enable=protected-access,invalid-name
    return cls_decorator


def flatten_space(space):
    """Flatten a space into a single ``Box``.
    This is equivalent to ``flatten()``, but operates on the space itself. The
    result always is a `Box` with flat boundaries. The box has exactly
    ``flatdim(space)`` dimensions. Flattening a sample of the original space
    has the same effect as taking a sample of the flattenend space.
    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.
    Example::
        >>> box = Box(0.0, 1.0, shape=(3, 4, 5))
        >>> box
        Box(3, 4, 5)
        >>> flatten_space(box)
        Box(60,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True
    Example that flattens a discrete space::
        >>> discrete = Discrete(5)
        >>> flatten_space(discrete)
        Box(5,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True
    Example that recursively flattens a dict::
        >>> space = Dict({"position": Discrete(2),
        ...               "velocity": Box(0, 1, shape=(2, 2))})
        >>> flatten_space(space)
        Box(6,)
        >>> flatten(space, space.sample()) in flatten_space(space)
        True
    """
    if isinstance(space, Box):
        return Box(space.low.flatten(), space.high.flatten())
    if isinstance(space, Discrete):
        return Box(low=0, high=1, shape=(space.n, ))
    if isinstance(space, Tuple):
        space = [flatten_space(s) for s in space.spaces]
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
        )
    if isinstance(space, Dict):
        space = [flatten_space(s) for s in space.spaces.values()]
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
        )
    if isinstance(space, MultiBinary):
        return Box(low=0, high=1, shape=(space.n, ))
    if isinstance(space, MultiDiscrete):
        return Box(
            low=np.zeros_like(space.nvec),
            high=space.nvec,
        )
    raise NotImplementedError


