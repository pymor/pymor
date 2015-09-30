# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip, chain
import os
import time
import weakref


try:
    from ipyparallel import Client, TimeoutError
    HAVE_IPYTHON = True
except ImportError:
    try:
        from IPython.parallel import Client, TimeoutError
        HAVE_IPYTHON = True
    except ImportError:
        HAVE_IPYTHON = False


from pymor.core.interfaces import BasicInterface, ImmutableInterface
from pymor.core.pickle import FunctionPicklingWrapper
from pymor.parallel.defaultimpl import WorkerPoolDefaultImplementations
from pymor.parallel.interfaces import WorkerPoolInterface, RemoteObjectInterface
from pymor.tools.counter import Counter


FunctionType = type(lambda x: x)


class new_ipcluster_pool(BasicInterface):

    def __init__(self, profile=None, cluster_id=None, num_engines=None, ipython_dir=None, min_wait=1, timeout=60):
        self.profile = profile
        self.cluster_id = cluster_id
        self.num_engines = num_engines
        self.ipython_dir = ipython_dir
        self.min_wait = min_wait
        self.timeout = timeout

    def __enter__(self):
        args = []
        if self.profile is not None:
            args.append('--profile=' + self.profile)
        if self.cluster_id is not None:
            args.append('--cluster-id=' + self.cluster_id)
        if self.num_engines is not None:
            args.append('--n=' + str(self.num_engines))
        if self.ipython_dir is not None:
            args.append('--ipython-dir=' + self.ipython_dir)
        cmd = ' '.join(['ipcluster start --daemonize'] + args)
        self.logger.info('Staring IPython cluster with "' + cmd + '"')
        os.system(cmd)

        num_engines, timeout = self.num_engines, self.timeout
        time.sleep(self.min_wait)
        waited = self.min_wait
        client = None
        while client is None:
            try:
                client = Client(profile=self.profile, cluster_id=self.cluster_id)
            except (IOError, TimeoutError):
                if waited >= self.timeout:
                    raise IOError('Could not connect to IPython cluster controller')
                if waited % 10 == 0:
                    self.logger.info('Waiting for controller to start ...')
                time.sleep(1)
                waited += 1

        if num_engines is None:
            while len(client) == 0 and waited < timeout:
                if waited % 10 == 0:
                    self.logger.info('Waiting for engines to start ...')
                time.sleep(1)
                waited += 1
            if len(client) == 0:
                raise IOError('IPython cluster engines failed to start')
            wait = min(waited, timeout - waited)
            if wait > 0:
                self.logger.info('Waiting {} more seconds for engines to start ...'.format(wait))
                time.sleep(wait)
        else:
            running = len(client)
            while running < num_engines and waited < timeout:
                if waited % 10 == 0:
                    self.logger.info('Waiting for {} of {} engines to start ...'
                                     .format(num_engines - running, num_engines))
                time.sleep(1)
                waited += 1
                running = len(client)
            running = len(client)
            if running < num_engines:
                raise IOError('{} of {} IPython cluster engines failed to start'
                              .format(num_engines - running, num_engines))
        client.close()

        self.pool = IPythonPool(profile=self.profile, cluster_id=self.cluster_id)
        return self.pool

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.client.close()
        args = []
        if self.profile is not None:
            args.append('--profile=' + self.profile)
        if self.cluster_id is not None:
            args.append('--cluster-id=' + self.cluster_id)
        if self.ipython_dir is not None:
            args.append('--ipython-dir=' + self.ipython_dir)
        cmd = ' '.join(['ipcluster stop'] + args)
        self.logger.info('Stopping IPython cluster with "' + cmd + '"')
        os.system(cmd)


class IPythonPool(WorkerPoolDefaultImplementations, WorkerPoolInterface):

    def __init__(self, num_engines=None, **kwargs):
        self.client = Client(**kwargs)
        if num_engines is not None:
            self.view = self.client[:num_engines]
        else:
            self.view = self.client[:]
        self.logger.info('Connected to {} engines'.format(len(self.view)))
        self.view.apply(_setup_worker, block=True)
        self._pushed_immutable_objects = {}
        self._remote_objects_created = Counter()

    def __len__(self):
        return len(self.view)

    def push(self, obj):
        if isinstance(obj, ImmutableInterface):
            uid = obj.uid
            if uid not in self._pushed_immutable_objects:
                remote_id = RemoteId(self._remote_objects_created.inc())
                self.view.apply_sync(_push_object, remote_id, obj)
                self._pushed_immutable_objects[uid] = (remote_id, 1)
            else:
                remote_id, ref_count = self._pushed_immutable_objects[uid]
                self._pushed_immutable_objects[uid] = (remote_id, ref_count + 1)
            return IPythonRemoteObject(self, remote_id, uid=uid)
        else:
            remote_id = RemoteId(self._remote_objects_created.inc())
            self.view.apply_sync(_push_object, remote_id, obj)
            return IPythonRemoteObject(self, remote_id)

    def _map_kwargs(self, kwargs):
        pushed_immutable_objects = self._pushed_immutable_objects
        return {k: (pushed_immutable_objects.get(v.uid, (v, 0))[0] if isinstance(v, ImmutableInterface) else
                    v.remote_id if isinstance(v, IPythonRemoteObject) else
                    FunctionPicklingWrapper(v) if isinstance(v, FunctionType) else
                    v)
                for k, v in kwargs.iteritems()}

    def apply(self, function, *args, **kwargs):
        function = FunctionPicklingWrapper(function)
        kwargs = self._map_kwargs(kwargs)
        return self.view.apply_sync(_worker_call_function, function, False, args, kwargs)

    def apply_only(self, function, worker, *args, **kwargs):
        view = self.client[worker]
        function = FunctionPicklingWrapper(function)
        kwargs = self._map_kwargs(kwargs)
        return view.apply_sync(_worker_call_function, function, False, args, kwargs)

    def map(self, function, *args, **kwargs):
        function = FunctionPicklingWrapper(function)
        kwargs = self._map_kwargs(kwargs)
        num_workers = len(self.view)
        chunks = _split_into_chunks(num_workers, *args)
        result = self.view.map_sync(_worker_call_function,
                                    *zip(*((function, True, a, kwargs) for a in izip(*chunks))))
        if isinstance(result[0][0], tuple):
            return tuple(list(x) for x in zip(*chain(*result)))
        else:
            return list(chain(*result))


class IPythonRemoteObject(RemoteObjectInterface):

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
                pool.view.apply(_remove_object, remote_id)
        else:
            pool.view.apply(_remove_object, self.remote_id)


class RemoteId(int):
    pass


def _worker_call_function(function, loop, args, kwargs):
    from pymor.core.pickle import FunctionPicklingWrapper
    global _remote_objects
    function = function.function
    kwargs = {k: (_remote_objects[v] if isinstance(v, RemoteId) else
                  v.function if isinstance(v, FunctionPicklingWrapper) else
                  v)
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


_remote_objects = {}


def _setup_worker():
    global _remote_objects
    _remote_objects.clear()


def _push_object(remote_id, obj):
    global _remote_objects
    _remote_objects[remote_id] = obj


def _remove_object(remote_id):
    global _remote_objects
    del _remote_objects[remote_id]
