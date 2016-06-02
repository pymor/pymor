# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from itertools import chain
import os
import time


try:
    from ipyparallel import Client, TimeoutError
    HAVE_IPYTHON = True
except ImportError:
    try:
        from IPython.parallel import Client, TimeoutError
        HAVE_IPYTHON = True
    except ImportError:
        HAVE_IPYTHON = False


from pymor.core.interfaces import BasicInterface
from pymor.parallel.basic import WorkerPoolBase
from pymor.tools.counter import Counter


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


class IPythonPool(WorkerPoolBase):

    def __init__(self, num_engines=None, **kwargs):
        super(IPythonPool, self).__init__()
        self.client = Client(**kwargs)
        if num_engines is not None:
            self.view = self.client[:num_engines]
        else:
            self.view = self.client[:]
        self.logger.info('Connected to {} engines'.format(len(self.view)))
        self.view.apply_sync(_setup_worker)
        self._remote_objects_created = Counter()

    def __len__(self):
        return len(self.view)

    def _push_object(self, obj):
        remote_id = RemoteId(self._remote_objects_created.inc())
        self.view.apply_sync(_push_object, remote_id, obj)
        return remote_id

    def _apply(self, function, *args, **kwargs):
        return self.view.apply_sync(_worker_call_function, function, False, args, kwargs)

    def _apply_only(self, function, worker, *args, **kwargs):
        view = self.client[worker]
        return view.apply_sync(_worker_call_function, function, False, args, kwargs)

    def _map(self, function, chunks, **kwargs):
        result = self.view.map_sync(_worker_call_function,
                                    *zip(*((function, True, a, kwargs) for a in zip(*chunks))))
        return list(chain(*result))

    def _remove_object(self, remote_id):
        self.view.apply(_remove_object, remote_id)


class RemoteId(int):
    pass


def _worker_call_function(function, loop, args, kwargs):
    global _remote_objects
    kwargs = {k: (_remote_objects[v] if isinstance(v, RemoteId) else  # NOQA
                  v)
              for k, v in kwargs.items()}
    if loop:
        return [function(*a, **kwargs) for a in zip(*args)]
    else:
        return function(*args, **kwargs)


def _setup_worker():
    global _remote_objects
    _remote_objects = {}


def _push_object(remote_id, obj):
    global _remote_objects
    _remote_objects[remote_id] = obj  # NOQA


def _remove_object(remote_id):
    global _remote_objects
    del _remote_objects[remote_id]  # NOQA
