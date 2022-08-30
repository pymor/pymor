# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from contextvars import ContextVar
import inspect

from pymor.core.defaults import defaults

import numpy as np


@defaults('seed_seq')
def init_rng(seed_seq=42):
    if not isinstance(seed_seq, np.random.SeedSequence):
        seed_seq = np.random.SeedSequence(seed_seq)
    # Store a new rng together with seed_seq, as the latter cannot be recovered from the
    # rng via a public interface (see https://github.com/numpy/numpy/issues/15322).
    # The first field is a flag to indicate whether the current _rng_state has been consumed
    # via get_rng. This is a safeguard to detect calls to get_rng in concurrent code
    # paths.
    _rng_state.set([False, np.random.default_rng(seed_seq), seed_seq])


def get_rng():
    rng_state = _get_rng_state()
    rng_state[0] = True
    _rng_state.set([False] + rng_state[1:])
    return rng_state[1]


class set_rng:

    def __init__(self, seed_seq):
        self.old_state = _rng_state.get(None)
        self.seed_seq = seed_seq
        self.set()

    def set(self):
        self._is_set = True
        init_rng(self.seed_seq)

    def reset(self):
        self._is_set = False
        _rng_state.set(self.old_state)

    def __enter__(self):
        if not self._is_set:
            self.set()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.reset()


def get_seed_seq():
    return _get_rng_state()[2]


def spawn_rng(f):
    seed_seq = get_seed_seq().spawn(1)[0]

    if inspect.iscoroutine(f):

        async def spawn_rng_wrapper():
            with set_rng(seed_seq):
                return await f

        return spawn_rng_wrapper()

    elif inspect.isfunction(f):

        return _SpawnRngWrapper(f, seed_seq)  # use a class to obtain something picklable

    else:
        raise TypeError


class _SpawnRngWrapper:
    def __init__(self, f, seed_seq):
        self.f, self.seed_seq = f, seed_seq

    def __call__(self, *args, **kwargs):
        with set_rng(self.seed_seq):
            return self.f(*args, **kwargs)


def _get_rng_state():
    try:
        rng_state = _rng_state.get()
    except LookupError:
        import warnings
        warnings.warn(
            'get_rng called but _rng_state not initialized. (Call spawn_rng when creating new thread.) '
            'Initializing a new RNG from the default seed. This may lead to correlated data.')
        init_rng()
        rng_state = _rng_state.get()
    if rng_state[0]:
        import warnings
        warnings.warn('You are using the same RNG in concurrent code paths.\n'
                      'This may lead to truly random, irreproducible behavior')
    return rng_state


_rng_state = ContextVar('_rng_state')
init_rng()
