# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from copy import deepcopy
from itertools import izip

from pymor.core.interfaces import ImmutableInterface
from pymor.parallel.interfaces import WorkerPoolInterface, RemoteObjectInterface


class DummyPool(WorkerPoolInterface):

    def __len__(self):
        return 1

    def push(self, obj):
        if isinstance(obj, ImmutableInterface):
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
        return {k: (v.obj if isinstance(v, DummyRemoteObject) else v) for k, v in kwargs.iteritems()}

    def apply(self, function, *args, **kwargs):
        kwargs = self._map_kwargs(kwargs)
        return [function(*args, **kwargs)]

    def apply_only(self, function, worker, *args, **kwargs):
        kwargs = self._map_kwargs(kwargs)
        return function(*args, **kwargs)

    def map(self, function, *args, **kwargs):
        kwargs = self._map_kwargs(kwargs)
        result = [function(*a, **kwargs) for a in izip(*args)]
        if isinstance(result[0], tuple):
            return zip(*result)
        else:
            return result


dummy_pool = DummyPool()


class DummyRemoteObject(RemoteObjectInterface):

    def __init__(self, obj):
        self.obj = obj

    def _remove(self):
        del self.obj
