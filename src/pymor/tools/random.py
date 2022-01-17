# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.defaults import defaults

import numpy as np


def get_random_state(random_state=None, seed=None):
    """Returns a |NumPy| :class:`~numpy.random.RandomState`.

    Parameters
    ----------
    random_state
        If specified, this state is returned.
    seed
        If specified, the seed to initialize a new random state.

    Returns
    -------
    Either the provided, a newly created or the default `RandomState`
    object.
    """
    assert random_state is None or seed is None
    if random_state is not None:
        return random_state
    elif seed is not None:
        return np.random.RandomState(seed)
    else:
        return default_random_state()


@defaults('seed')
def default_random_state(seed=42):
    """Returns the default |NumPy| :class:`~numpy.random.RandomState`.

    Parameters
    ----------
    seed
        Seed to use for initializing the random state.

    Returns
    -------
    The default `RandomState` object.
    """
    global _default_random_state

    if _default_random_state is None:
        _default_random_state = np.random.RandomState(seed)

    return _default_random_state


_default_random_state = None
