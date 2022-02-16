# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import atexit

from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.parallel.dummy import dummy_pool


@defaults('ipython_num_engines', 'ipython_profile', 'allow_mpi')
def new_parallel_pool(ipython_num_engines=None, ipython_profile=None, allow_mpi=True):
    """Creates a new default |WorkerPool|.

    If `ipython_num_engines` or `ipython_profile` is provided as an argument or set as
    a |default|, an :class:`~pymor.parallel.ipython.IPythonPool` |WorkerPool| will
    be created using the given parameters via the `ipcluster` script.

    Otherwise, when `allow_mpi` is `True` and an MPI parallel run is detected,
    an :class:`~pymor.parallel.mpi.MPIPool` |WorkerPool| will be created.

    Otherwise, a sequential run is assumed and
    :attr:`pymor.parallel.dummy.dummy_pool <pymor.parallel.dummy.DummyPool>`
    is returned.
    """
    global _pool
    if _pool:
        logger = getLogger('pymor.parallel.default.new_parallel_pool')
        logger.warning('new_parallel_pool already called; returning old pool (this might not be what you want).')
        return _pool[1]
    if ipython_num_engines or ipython_profile:
        from pymor.parallel.ipython import new_ipcluster_pool
        nip = new_ipcluster_pool(profile=ipython_profile, num_engines=ipython_num_engines)
        pool = nip.__enter__()
        _pool = ('ipython', pool, nip)
        return pool
    elif allow_mpi:
        from pymor.tools import mpi
        if mpi.parallel:
            from pymor.parallel.mpi import MPIPool
            pool = MPIPool()
            _pool = ('mpi', pool)
            return pool
        else:
            _pool = ('dummy', dummy_pool)
            return dummy_pool
    else:
        _pool = ('dummy', dummy_pool)
        return dummy_pool


_pool = None


@atexit.register
def _cleanup():
    global _pool
    if _pool and _pool[0] == 'ipython':
        _pool[2].__exit__(None, None, None)
    _pool = None
