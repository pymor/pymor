# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip, chain
import weakref

from IPython.parallel import Client

from pymor.core.interfaces import ImmutableInterface
from pymor.core.pickle import dumps, dumps_function, PicklingError
from pymor.parallel.interfaces import WorkerPoolInterface
from pymor.tools.counter import Counter


def _worker_call_function(function, loop, args, kwargs):
    from pymor.core.pickle import loads, loads_function
    global remote_objects
    if isinstance(function, tuple):
        function = loads_function(function[0])
    else:
        function = loads(function)
    kwargs = {k: remote_objects[v.key] if isinstance(v, RemoteObject) else v
              for k, v in kwargs.iteritems()}
    if loop:
        return [function(*a, **kwargs) for a in izip(*args)]
    else:
        return function(*args, **kwargs)


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


remote_objects = {}


def _setup_worker():
    global remote_objects
    remote_objects.clear()


def _distribute_objects(objs):
    global remote_objects
    remote_objects.update(objs)


def _remove_objects(ids):
    global remote_objects
    for i in ids:
        del remote_objects[i]


class RemoteObject(object):

    def __init__(self, key):
        self.key = key


class IPythonPool(WorkerPoolInterface):

    def __init__(self):
        self.client = Client()
        self.view = self.client[:]
        self.view.apply(_setup_worker, block=True)
        self._distributed_immutable_objects = {}
        self._remote_objects_created = Counter()

    def __len__(self):
        return len(self.client)

    def distribute(self, *args):
        return IPythonPool.DistributedObjectManager(self, *args)

    def _pickle_function(self, function):
        if hasattr(function, '__file__'):
            try:
                function = dumps(function)
            except PicklingError:
                function = (dumps_function(function),)
        else:
            function = (dumps_function(function),)
        return function

    def _map_kwargs(self, kwargs):
        distributed_immutable_objects = self._distributed_immutable_objects
        return {k: distributed_immutable_objects.get(v.uid, v) if isinstance(v, ImmutableInterface) else v
                for k, v in kwargs.iteritems()}

    def apply(self, function, *args, **kwargs):
        function = self._pickle_function(function)
        kwargs = self._map_kwargs(kwargs)
        return self.view.apply_sync(_worker_call_function, function, False, args, kwargs)

    def apply_only(self, function, worker, *args, **kwargs):
        view = self.client[worker]
        function = self._pickle_function(function)
        kwargs = self._map_kwargs(kwargs)
        return view.apply_sync(_worker_call_function, function, False, args, kwargs)

    def map(self, function, *args, **kwargs):
        function = self._pickle_function(function)
        kwargs = self._map_kwargs(kwargs)
        num_workers = len(self.view)
        chunks = _split_into_chunks(num_workers, *args)
        result = self.view.map_sync(_worker_call_function,
                                    *zip(*((function, True, a, kwargs) for a in izip(*chunks))))
        if isinstance(result[0][0], tuple):
            return tuple(list(x) for x in zip(*chain(*result)))
        else:
            return list(chain(*result))

    class DistributedObjectManager(object):

        def __init__(self, pool, *args):
            self.pool = weakref.ref(pool)
            self.objs = args

        def __enter__(self):
            pool = self.pool()
            objects_to_distribute = {}
            self.remote_objects_to_remove = []
            self.distributed_immutable_objects_to_remove = []

            def process_obj(o):
                if isinstance(o, ImmutableInterface):
                    if o.uid not in pool._distributed_immutable_objects:
                        remote_id = pool._remote_objects_created.inc()
                        objects_to_distribute[remote_id] = o
                        pool._distributed_immutable_objects[o.uid] = RemoteObject(remote_id)
                        self.remote_objects_to_remove.append(remote_id)
                        self.distributed_immutable_objects_to_remove.append(o.uid)
                    return pool._distributed_immutable_objects[o.uid]
                else:
                    remote_id = pool._remote_objects_created.inc()
                    objects_to_distribute[remote_id] = o
                    self.remote_objects_to_remove.append(remote_id)
                    return RemoteObject(remote_id)

            remote_objects = tuple(process_obj(o) for o in self.objs)
            pool.view.apply_sync(_distribute_objects, objects_to_distribute)

            # release local refecrences to distributed objects
            del self.objs

            return remote_objects

        def __exit__(self, exc_type, exc_val, exc_tb):
            pool = self.pool()
            pool.view.apply(_remove_objects, self.remote_objects_to_remove)
            for uid in self.distributed_immutable_objects_to_remove:
                del pool._distributed_immutable_objects[uid]
            return False
