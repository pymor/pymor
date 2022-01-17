# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""This module contains a base class for implementing WorkerPool."""

import weakref

from pymor.core.base import ImmutableObject
from pymor.parallel.interface import WorkerPool, RemoteObject


class WorkerPoolDefaultImplementations:

    def scatter_array(self, U, copy=True):
        slice_len = len(U) // len(self) + (1 if len(U) % len(self) else 0)
        if copy:
            slices = []
            for i in range(len(self)):
                slices.append(U[i*slice_len:min((i+1)*slice_len, len(U))].copy())
        else:
            slices = [U.empty() for _ in range(len(self))]
            for s in slices:
                s.append(U[:min(slice_len, len(U))], remove_from_other=True)
        remote_U = self.push(U.empty())
        del U
        self.map(_append_array_slice, slices, U=remote_U)
        return remote_U

    def scatter_list(self, l):
        slice_len = len(l) // len(self) + (1 if len(l) % len(self) else 0)
        slices = []
        for i in range(len(self)):
            slices.append(l[i*slice_len:(i+1)*slice_len])
        del l
        remote_l = self.push([])
        self.map(_append_list_slice, slices, l=remote_l)
        return remote_l


class WorkerPoolBase(WorkerPoolDefaultImplementations, WorkerPool):

    def __init__(self):
        self._pushed_immutable_objects = {}

    def push(self, obj):
        if isinstance(obj, ImmutableObject):
            uid = obj.uid
            if uid not in self._pushed_immutable_objects:
                remote_id = self._push_object(obj)
                self._pushed_immutable_objects[uid] = (remote_id, 1)
            else:
                remote_id, ref_count = self._pushed_immutable_objects[uid]
                self._pushed_immutable_objects[uid] = (remote_id, ref_count + 1)
            return GenericRemoteObject(self, remote_id, uid=uid)
        else:
            remote_id = self._push_object(obj)
            return GenericRemoteObject(self, remote_id)

    def _map_kwargs(self, kwargs):
        pushed_immutable_objects = self._pushed_immutable_objects
        return {k: (pushed_immutable_objects.get(v.uid, (v, 0))[0] if isinstance(v, ImmutableObject) else
                    v.remote_id if isinstance(v, RemoteObject) else
                    v)
                for k, v in kwargs.items()}

    def apply(self, function, *args, **kwargs):
        kwargs = self._map_kwargs(kwargs)
        return self._apply(function, *args, **kwargs)

    def apply_only(self, function, worker, *args, **kwargs):
        kwargs = self._map_kwargs(kwargs)
        return self._apply_only(function, worker, *args, **kwargs)

    def map(self, function, *args, **kwargs):
        kwargs = self._map_kwargs(kwargs)
        chunks = self._split_into_chunks(len(self), *args)
        return self._map(function, chunks, **kwargs)

    def _split_into_chunks(self, count, *args):
        lens = list(map(len, args))
        min_len = min(lens)
        max_len = max(lens)
        assert min_len == max_len
        chunk_size = max_len // count + (1 if max_len % count > 0 else 0)

        def split_arg(arg):
            for _ in range(count):
                chunk, arg = arg[:chunk_size], arg[chunk_size:]
                yield chunk
        from itertools import chain
        for arg in args:
            assert list(chain(*split_arg(arg))) == arg
        chunks = tuple(list(split_arg(arg)) for arg in args)
        return chunks


class GenericRemoteObject(RemoteObject):

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
                pool._remove_object(remote_id)
        else:
            pool._remove_object(self.remote_id)


def _append_array_slice(s, U=None):
    U.append(s, remove_from_other=True)


def _append_list_slice(s, l=None):
    l.extend(s)
