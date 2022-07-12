# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
config.require('IPYTHON')


from itertools import chain
import os
import time

from pymor.core.base import BasicObject
from pymor.core import defaults
from pymor.parallel.basic import WorkerPoolBase
from pymor.tools.counter import Counter


try:
    from ipyparallel import Client, TimeoutError
except ImportError:
    from IPython.parallel import Client, TimeoutError


class new_ipcluster_pool(BasicObject):
    """Create a new IPython parallel cluster and connect to it.

    This context manager can be used to create an :class:`IPythonPool`
    |WorkerPool|. When entering the context a new IPython cluster is
    created using the `ipcluster` script and an :class:`IPythonPool`
    is instantiated for the newly created cluster. When leaving
    the context the cluster is shut down.

    Parameters
    ----------
    profile
        Passed as `--profile` parameter to the `ipcluster` script.
    cluster_id
        Passed as `--cluster-id` parameter to the `ipcluster` script.
    nun_engines
        Passed as `--n` parameter to the `ipcluster` script.
    ipython_dir
        Passed as `--ipython-dir` parameter to the `ipcluster` script.
    min_wait
        Wait at least this many seconds before trying to connect to the
        new cluster.
    timeout
        Wait at most this many seconds for all Ipython cluster engines to
        become available.
    """

    def __init__(self, profile=None, cluster_id=None, num_engines=None, ipython_dir=None, min_wait=1, timeout=60):
        self.__auto_init(locals())

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
            except (IOError, TimeoutError) as e:
                if waited >= self.timeout:
                    raise IOError('Could not connect to IPython cluster controller') from e
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
                self.logger.info(f'Waiting {wait} more seconds for engines to start ...')
                time.sleep(wait)
        else:
            running = len(client)
            while running < num_engines and waited < timeout:
                if waited % 10 == 0:
                    self.logger.info(f'Waiting for {num_engines-running} of {num_engines} engines to start ...')
                time.sleep(1)
                waited += 1
                running = len(client)
            running = len(client)
            if running < num_engines:
                raise IOError(f'{num_engines-running} of {num_engines} IPython cluster engines failed to start')
        # make sure all (potential) engines are in the same cwd, so they can import the same code
        client[:].apply_sync(os.chdir, os.getcwd())
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
    """|WorkerPool| based on the IPython parallel computing features.

    Parameters
    ----------
    num_engines
        Number of IPython engines to use. If `None`, all available
        engines are used.
    kwargs
        Keyword arguments used to instantiate the IPython cluster client.
    """

    _updated_defaults = 0

    def __init__(self, num_engines=None, **kwargs):
        super().__init__()
        self.client = Client(**kwargs)
        if num_engines is not None:
            self.view = self.client[:num_engines]
        else:
            self.view = self.client[:]
        self.logger.info(f'Connected to {len(self.view)} engines')
        self.view.map_sync(_setup_worker, range(len(self.view)))
        self._remote_objects_created = Counter()

        if defaults.defaults_changes() > 0:
            self._update_defaults()

    def __len__(self):
        return len(self.view)

    def _push_object(self, obj):
        remote_id = RemoteId(self._remote_objects_created.inc())
        self.view.apply_sync(_push_object, remote_id, obj)
        return remote_id

    def _apply(self, function, *args, **kwargs):
        if defaults.defaults_changes() > self._updated_defaults:
            self._update_defaults()
        return self.view.apply_sync(_worker_call_function, function, False, args, kwargs)

    def _apply_only(self, function, worker, *args, **kwargs):
        if defaults.defaults_changes() > self._updated_defaults:
            self._update_defaults()
        view = self.client[int(worker)]
        return view.apply_sync(_worker_call_function, function, False, args, kwargs)

    def _map(self, function, chunks, **kwargs):
        if defaults.defaults_changes() > self._updated_defaults:
            self._update_defaults()
        result = self.view.map_sync(_worker_call_function,
                                    *zip(*((function, True, a, kwargs) for a in zip(*chunks))))
        return list(chain(*result))

    def _remove_object(self, remote_id):
        self.view.apply(_remove_object, remote_id)

    def _update_defaults(self):
        self._updated_defaults = defaults.defaults_changes()
        self._apply(defaults.set_defaults, defaults.get_defaults(user=True, file=True, code=False))


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


def _setup_worker(worker_id):
    global _remote_objects
    _remote_objects = {}
    # ensure that each worker starts with a different RandomState
    from pymor.tools import random
    import numpy as np
    state = random.default_random_state()
    new_state = np.random.RandomState(state.randint(0, 2**16) + worker_id)
    random._default_random_state = new_state


def _push_object(remote_id, obj):
    global _remote_objects
    _remote_objects[remote_id] = obj  # NOQA


def _remove_object(remote_id):
    global _remote_objects
    del _remote_objects[remote_id]  # NOQA
