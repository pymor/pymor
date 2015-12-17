# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import chain
import weakref


from pymor.core.interfaces import ImmutableInterface
from pymor.core.pickle import FunctionPicklingWrapper
from pymor.parallel.defaultimpl import WorkerPoolDefaultImplementations
from pymor.parallel.interfaces import WorkerPoolInterface, RemoteObjectInterface
from pymor.tools.counter import Counter
from pymor.tools import mpi


FunctionType = type(lambda x: x)


class MPIPool(WorkerPoolDefaultImplementations, WorkerPoolInterface):

    def __init__(self):
        self.logger.info('Connected to {} ranks'.format(mpi.size))
        self._pushed_immutable_objects = {}
        self._remote_objects_created = Counter()
        self._payload = mpi.call(mpi.function_call_manage, _setup_worker)

    def __del__(self):
        mpi.call(mpi.remove_object, self._payload)

    def __len__(self):
        return mpi.size

    def push(self, obj):
        if isinstance(obj, ImmutableInterface):
            uid = obj.uid
            if uid not in self._pushed_immutable_objects:
                remote_id = mpi.call(mpi.function_call_manage, _push_object, obj)
                self._pushed_immutable_objects[uid] = (remote_id, 1)
            else:
                remote_id, ref_count = self._pushed_immutable_objects[uid]
                self._pushed_immutable_objects[uid] = (remote_id, ref_count + 1)
            return MPIRemoteObject(self, remote_id, uid=uid)
        else:
            remote_id = mpi.call(mpi.function_call_manage, _push_object, obj)
            return MPIRemoteObject(self, remote_id)

    def _map_kwargs(self, kwargs):
        pushed_immutable_objects = self._pushed_immutable_objects
        return {k: (pushed_immutable_objects.get(v.uid, (v, 0))[0] if isinstance(v, ImmutableInterface) else
                    v.remote_id if isinstance(v, MPIRemoteObject) else
                    v)
                for k, v in kwargs.iteritems()}

    def apply(self, function, *args, **kwargs):
        function = FunctionPicklingWrapper(function)
        kwargs = self._map_kwargs(kwargs)
        return mpi.call(mpi.function_call, _worker_call_function, function, *args, **kwargs)

    def apply_only(self, function, worker, *args, **kwargs):
        function = FunctionPicklingWrapper(function)
        kwargs = self._map_kwargs(kwargs)
        payload = mpi.get_object(self._payload)
        payload[0] = (function, args, kwargs)
        try:
            result = mpi.call(mpi.function_call, _single_worker_call_function, self._payload, worker)
        finally:
            payload[0] = None
        return result

    def map(self, function, *args, **kwargs):
        function = FunctionPicklingWrapper(function)
        kwargs = self._map_kwargs(kwargs)
        payload = mpi.get_object(self._payload)
        payload[0] = _split_into_chunks(mpi.size, *args)
        try:
            result = mpi.call(mpi.function_call, _worker_map_function, self._payload, function, **kwargs)
        finally:
            payload[0] = None
        return result


class MPIRemoteObject(RemoteObjectInterface):

    def __init__(self, pool, remote_id, uid=None):
        self.pool = weakref.ref(pool)
        self.remote_id = remote_id
        self.uid = uid

    def _remove(self):
        pool = self.pool()
        if self.uid is not None:
            remote_id, ref_count = pool._pushed_immutable_objects.pop(self.uid)
            if ref_count > 1:
                pool._pushed_immutable_objects[self.remote_id] = (remote_id, ref_count - 1)
            else:
                mpi.call(mpi.remove_object, remote_id)
        else:
            mpi.call(mpi.remove_object, self.remote_id)


def _worker_call_function(function, *args, **kwargs):
    result = function.function(*args, **kwargs)
    return mpi.comm.gather(result, root=0)


def _single_worker_call_function(payload, worker):
    if mpi.rank0:
        if worker == 0:
            function, args, kwargs = payload[0]
            return mpi.function_call(function.function, *args, **kwargs)
        else:
            mpi.comm.send(payload[0], dst=worker)
            return mpi.comm.recv(source=worker)
    else:
        if mpi.rank != worker:
            return
        (function, args, kwargs) = mpi.comm.recv(source=0)
        retval = mpi.function_call(function.function, *args, **kwargs)
        mpi.comm.send(retval, dst=0)


def _worker_map_function(payload, function, **kwargs):
    function = function.function

    if mpi.rank0:
        args = zip(*payload[0])
    else:
        args = None
    args = zip(*mpi.comm.scatter(args, root=0))

    result = [mpi.function_call(function, *a, **kwargs) for a in args]
    result = mpi.comm.gather(result, root=0)

    if mpi.rank0:
        return list(chain(*result))


def _split_into_chunks(count, *args):
    lens = map(len, args)
    min_len = min(lens)
    max_len = max(lens)
    assert min_len == max_len
    chunk_size = max_len // count + (1 if max_len % count > 0 else 0)

    def split_arg(arg):
        while arg:
            chunk, arg = arg[:chunk_size], arg[chunk_size:]
            yield chunk
    chunks = tuple(list(split_arg(arg)) for arg in args)
    return chunks


def _setup_worker():
    return [None]


def _push_object(obj):
    return obj
