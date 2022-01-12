# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import chain
import os


from pymor.parallel.basic import WorkerPoolBase
from pymor.tools import mpi


class MPIPool(WorkerPoolBase):
    """|WorkerPool| based pyMOR's MPI :mod:`event loop <pymor.tools.mpi>`."""

    def __init__(self):
        super().__init__()
        self.logger.info(f'Connected to {mpi.size} ranks')
        self._payload = mpi.call(mpi.function_call_manage, _setup_worker)
        self._apply(os.chdir, os.getcwd())

    def __del__(self):
        mpi.call(mpi.remove_object, self._payload)

    def __len__(self):
        return mpi.size

    def _push_object(self, obj):
        return mpi.call(mpi.function_call_manage, _push_object, obj)

    def _apply(self, function, *args, **kwargs):
        return mpi.call(mpi.function_call, _worker_call_function, function, *args, **kwargs)

    def _apply_only(self, function, worker, *args, **kwargs):
        payload = mpi.get_object(self._payload)
        payload[0] = (function, args, kwargs)
        try:
            result = mpi.call(mpi.function_call, _single_worker_call_function, self._payload, worker)
        finally:
            payload[0] = None
        return result

    def _map(self, function, chunks, **kwargs):
        payload = mpi.get_object(self._payload)
        payload[0] = chunks
        try:
            result = mpi.call(mpi.function_call, _worker_map_function, self._payload, function, **kwargs)
        finally:
            payload[0] = None
        return result

    def _remove_object(self, remote_id):
        mpi.call(mpi.remove_object, remote_id)


def _worker_call_function(function, *args, **kwargs):
    result = function(*args, **kwargs)
    return mpi.comm.gather(result, root=0)


def _single_worker_call_function(payload, worker):
    if mpi.rank0:
        if worker == 0:
            function, args, kwargs = payload[0]
            return mpi.function_call(function, *args, **kwargs)
        else:
            mpi.comm.send(payload[0], dest=worker)
            return mpi.comm.recv(source=worker)
    else:
        if mpi.rank != worker:
            return
        (function, args, kwargs) = mpi.comm.recv(source=0)
        retval = mpi.function_call(function, *args, **kwargs)
        mpi.comm.send(retval, dest=0)


def _worker_map_function(payload, function, **kwargs):

    if mpi.rank0:
        args = list(zip(*payload[0]))
    else:
        args = None
    args = zip(*mpi.comm.scatter(args, root=0))

    result = [mpi.function_call(function, *a, **kwargs) for a in args]
    result = mpi.comm.gather(result, root=0)

    if mpi.rank0:
        return list(chain(*result))


def _setup_worker():
    # ensure that each worker starts with a different RandomState
    if not mpi.rank0:
        from pymor.tools import random
        import numpy as np
        state = random.default_random_state()
        new_state = np.random.RandomState(state.randint(0, 2**16) + mpi.rank)
        random._default_random_state = new_state
    return [None]


def _push_object(obj):
    return obj
