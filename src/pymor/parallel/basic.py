# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module contains a base class for implementing WorkerPoolInterface."""

from __future__ import absolute_import, division, print_function

import weakref

from pymor.core.interfaces import ImmutableInterface
from pymor.core.pickle import FunctionPicklingWrapper
from pymor.parallel.defaultimpl import WorkerPoolDefaultImplementations
from pymor.parallel.interfaces import WorkerPoolInterface, RemoteObjectInterface


class WorkerPoolBase(WorkerPoolDefaultImplementations, WorkerPoolInterface):

    def __init__(self):
        self._pushed_immutable_objects = {}

    def push(self, obj):
        if isinstance(obj, ImmutableInterface):
            uid = obj.uid
            if uid not in self._pushed_immutable_objects:
                remote_id = self._push_object(obj)
                self._pushed_immutable_objects[uid] = (remote_id, 1)
            else:
                remote_id, ref_count = self._pushed_immutable_objects[uid]
                self._pushed_immutable_objects[uid] = (remote_id, ref_count + 1)
            return RemoteObject(self, remote_id, uid=uid)
        else:
            remote_id = self._push_object(obj)
            return RemoteObject(self, remote_id)

    def _map_kwargs(self, kwargs):
        pushed_immutable_objects = self._pushed_immutable_objects
        return {k: (pushed_immutable_objects.get(v.uid, (v, 0))[0] if isinstance(v, ImmutableInterface) else
                    v.remote_id if isinstance(v, RemoteObject) else
                    v)
                for k, v in kwargs.iteritems()}

    def apply(self, function, *args, **kwargs):
        function = FunctionPicklingWrapper(function)
        kwargs = self._map_kwargs(kwargs)
        return self._apply(function, *args, **kwargs)

    def apply_only(self, function, worker, *args, **kwargs):
        function = FunctionPicklingWrapper(function)
        kwargs = self._map_kwargs(kwargs)
        return self._apply_only(function, worker, *args, **kwargs)

    def map(self, function, *args, **kwargs):
        function = FunctionPicklingWrapper(function)
        kwargs = self._map_kwargs(kwargs)
        chunks = self._split_into_chunks(len(self), *args)
        return self._map(function, chunks, **kwargs)

    def _split_into_chunks(self, count, *args):
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


class RemoteObject(RemoteObjectInterface):

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
