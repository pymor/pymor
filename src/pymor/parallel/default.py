# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import atexit

from pymor.core.defaults import defaults
from pymor.parallel.dummy import dummy_pool


@defaults('ipython_num_engines', 'ipython_profile', 'allow_mpi')
def new_parallel_pool(ipython_num_engines=None, ipython_profile=None, allow_mpi=True):
    """Creates a new default |WorkerPool|.

    If `ipython_num_engines` or `ipython_profile` is given by the user or set as a
    |default|, an :class:`~pymor.parallel.ipython.IPythonPool` |WorkerPool| will
    be created using the given parameters via the `ipcluster` script.

    Otherwise, when `allow_mpi` is `True` and an MPI parallel run is detected,
    an :class:`~pymor.parallel.mpi.MPIPool` |WorkerPool| will be created.

    Otherwise, a sequential run is assumed and :attr:`pymor.parallel.dummy.dummy_pool`
    is returned.
    """

    if ipython_num_engines or ipython_profile:
        from pymor.parallel.ipython import new_ipcluster_pool
        nip = new_ipcluster_pool(profile=ipython_profile, num_engines=ipython_num_engines)
        pool = nip.__enter__()
        atexit.register(lambda: nip.__exit__(None, None, None))
        return pool
    elif allow_mpi:
        from pymor.tools import mpi
        if mpi.parallel:
            from pymor.parallel.mpi import MPIPool
            return MPIPool()
        else:
            return dummy_pool
    else:
        return dummy_pool
