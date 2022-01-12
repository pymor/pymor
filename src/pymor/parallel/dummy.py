# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from copy import deepcopy

from pymor.core.base import ImmutableObject
from pymor.parallel.interface import WorkerPool, RemoteObject


class DummyPool(WorkerPool):

    def __len__(self):
        return 1

    def push(self, obj):
        if isinstance(obj, ImmutableObject):
            return DummyRemoteObject(obj)
        else:
            return DummyRemoteObject(deepcopy(obj))  # ensure we make a deep copy of the data

    def scatter_array(self, U, copy=True):
        if copy:
            U = U.copy()
        return DummyRemoteObject(U)

    def scatter_list(self, l):
        l = list(l)
        return DummyRemoteObject(l)

    def _map_kwargs(self, kwargs):
        return {k: (v.obj if isinstance(v, DummyRemoteObject) else v) for k, v in kwargs.items()}

    def apply(self, function, *args, **kwargs):
        kwargs = self._map_kwargs(kwargs)
        return [function(*args, **kwargs)]

    def apply_only(self, function, worker, *args, **kwargs):
        kwargs = self._map_kwargs(kwargs)
        return function(*args, **kwargs)

    def map(self, function, *args, **kwargs):
        kwargs = self._map_kwargs(kwargs)
        result = [function(*a, **kwargs) for a in zip(*args)]
        return result

    def __bool__(self):
        return False


dummy_pool = DummyPool()


class DummyRemoteObject(RemoteObject):

    def __init__(self, obj):
        self.obj = obj

    def _remove(self):
        del self.obj
