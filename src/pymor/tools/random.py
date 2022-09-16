# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Methods for managing random state in pyMOR.

Many algorithms potentially depend directly or indirectly on randomness.
To ensure reproducible execution of pyMOR code without having to pass around
a random number generator object everywhere, pyMOR manages a global |RNG|
object. This object is initialized automatically from a configurable |default|
random seed during startup and can be obtained by calling :func:`get_rng`.
The returned object is a subclass of :class:`numpy.random.Generator` and
inherits all its sampling methods.

To locally reset the global |RNG| in order to deterministically sample random
numbers independently of previously executed code, a new |RNG| can be created
via :func:`new_rng` and installed by using it as a context manager. For
instance, to sample a deterministic initialization vector for an iterative
algorithm we can write:

.. code:: python

    with new_rng(12345):
        U0 = some_operator.source.random()

Using a single global random state can lead to either non-deterministic or
correlated behavior in parallel or asynchronous code. :func:`get_rng` takes
provisions to detect such situations and issue a warning. In such cases
:func:`spawn_rng` needs to be called on the entry points of concurrent code
paths to ensure the desired behavior. For an advanced example, see
:mod:`pymor.algorithms.hapod`.
"""

from contextvars import ContextVar
import inspect

from pymor.core.defaults import defaults

import numpy as np


def get_rng():
    """Returns the current globally installed :class:`random number generator <RNG>`."""
    rng_state = _get_rng_state()
    rng_state[0] = True
    _rng_state.set([False] + rng_state[1:])
    return rng_state[1]


@defaults('seed_seq')
def new_rng(seed_seq=42):
    """Creates a new |RNG| and returns it.

    Parameters
    ----------
    seed_seq
        Entropy to seed the generator with. Either a :class:`~numpy.random.SeedSequence`
        or an `int` or list of `ints` from which the :class:`~numpy.random.SeedSequence`
        will be created. If `None`, entropy is sampled from the operating system.

    Returns
    -------
    The newly created |RNG|.
    """
    if not isinstance(seed_seq, np.random.SeedSequence):
        seed_seq = np.random.SeedSequence(seed_seq)
    return RNG(seed_seq)


class RNG(np.random.Generator):
    """Random number generator.

    This class inherits from :class:`np.random.Generator` and inherits all its sampling
    methods. Further, the class can be used as a context manager, which upon entry
    installs the RNG as pyMOR's global RNG that is returned from :func:`get_rng`.
    When the context is left, the previous global RNG is installed again.

    When using a context manager is not feasible, i.e. in an interactive workflow, this
    functionality can be accessed via the :meth:`~RNG.install` and :meth:`~RNG:uninstall`
    methods.

    A new instance of this class should be obtained using :func:`new_rng`.

    Parameters
    ----------
    seed_seq
        A :class:`~numpy.random.SeedSequence` to initialized the RNG with.
    """

    def __init__(self, seed_seq):
        self.old_state = _rng_state.get(None)
        super().__init__(np.random.default_rng(seed_seq).bit_generator)
        self._seed_seq = seed_seq

    def install(self):
        """Installs the generator as pyMOR's global random generator."""
        # Store a new rng together with seed_seq, as the latter cannot be recovered from the
        # rng via a public interface (see https://github.com/numpy/numpy/issues/15322).
        # The first field is a flag to indicate whether the current _rng_state has been consumed
        # via get_rng. This is a safeguard to detect calls to get_rng in concurrent code
        # paths.
        _rng_state.set([False, self, self._seed_seq])

    def uninstall(self):
        """Restores the previously set global random generator."""
        _rng_state.set(self.old_state)

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.uninstall()


def get_seed_seq():
    """Returns :class:`~np.random.SeedSequence` of the current global |RNG|.

    This function returns the :class:`~np.random.SeedSequence` with which pyMOR's
    currently installed global |RNG| has been initialized. The returned instance can
    be used to deterministically create a new :class:`~np.random.SeedSequence` via
    the :meth:`~np.random.SeedSequence.spawn` method, which then can be used to
    initialize a new random generator in external library code or concurrent code
    paths.
    """
    return _get_rng_state()[2]


def spawn_rng(f):
    """Wraps a function or coroutine to create a new |RNG| in concurrent code paths.

    Calling this function on a function or coroutine object creates a wrapper which
    will execute the wrapped function with a new globally installed |RNG|. This
    ensures that random numbers in concurrent code paths (:mod:`threads <threading>`,
    :mod:`multiprocessing`, :mod:`asyncio`) are deterministically generated yet
    uncorrelated.

    .. warning::
        If the control flow within a single code path depends on communication events
        with concurrent code, e.g., the order in which some parallel jobs finish,
        deterministic behavior can no longer be guaranteed by just using :func:`spawn_rng`.
        In such cases, the code additionally has to ensure that random numbers are sampled
        independently of the communication order.

    Parameters
    ----------
    f
        The function or coroutine to wrap.

    Returns
    -------
    The wrapped function or coroutine.
    """
    seed_seq = get_seed_seq().spawn(1)[0]

    if inspect.iscoroutine(f):

        async def spawn_rng_wrapper():
            with new_rng(seed_seq):
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
        with new_rng(self.seed_seq):
            return self.f(*args, **kwargs)


def _get_rng_state():
    try:
        rng_state = _rng_state.get()
    except LookupError:
        import warnings
        warnings.warn(
            'get_rng called but _rng_state not initialized. (Call spawn_rng when creating new thread.) '
            'Initializing a new RNG from the default seed. This may lead to correlated data.')
        new_rng().install()
        rng_state = _rng_state.get()
    if rng_state[0]:
        import warnings
        warnings.warn('You are using the same RNG in concurrent code paths.\n'
                      'This may lead to truly random, irreproducible behavior')
    return rng_state


_rng_state = ContextVar('_rng_state')
new_rng().install()
