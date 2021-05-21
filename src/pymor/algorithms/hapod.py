# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import asyncio
from math import ceil
import numpy as np
from queue import LifoQueue
from threading import Lock

from pymor.algorithms.pod import pod
from pymor.core.base import BasicObject, abstractmethod
from pymor.core.logger import getLogger


class Tree(BasicObject):
    """A rooted tree."""

    root = 0

    @abstractmethod
    def children(self, node):
        pass

    # Allows trees to provide their dept, since the generic depth() computation below easily exceeds Python's recursion
    # limit (i.e. the IncHAPODTree for a vectorarray of length larger than `import sys; sys.getrecursionlimit()`).
    def _depth(self):
        return None

    @property
    def depth(self):
        d = self._depth()
        if d:
            return d
        else:
            def get_depth(node):
                children = self.children(node)
                if children:
                    return 1 + max(get_depth(c) for c in children)
                else:
                    return 1
            return get_depth(self.root)

    def is_leaf(self, node):
        return not self.children(node)


class IncHAPODTree(Tree):

    def __init__(self, steps):
        self.steps = steps
        self.root = steps

    def children(self, node):
        if node < 0:
            return ()
        elif node == 1:
            return (-1,)
        else:
            return (node - 1, -node)

    def _depth(self):
        return self.steps + 1


class DistHAPODTree(Tree):

    def __init__(self, slices):
        self.root = slices

    def children(self, node):
        return tuple(range(self.root)) if node == self.root else ()

    def _depth(self):
        return 2


def default_pod_method(U, eps, is_root_node, product):
    return pod(U, atol=0., rtol=0.,
               l2_err=eps, product=product,
               orth_tol=None if is_root_node else np.inf)


def hapod(tree, snapshots, local_eps, product=None, pod_method=default_pod_method,
          executor=None, eval_snapshots_in_executor=False):
    """Compute the Hierarchical Approximate POD.

    This is an implementation of the HAPOD algorithm from :cite:`HLR18`.

    Parameters
    ----------
    tree
        A :class:`Tree` defining the worker topology.
    snapshots
        A mapping `snapshots(node)` returning for each leaf node the
        associated snapshot vectors.
    local_eps
        A mapping `local_eps(node, snap_count, num_vecs)` assigning
        to each tree node `node` an l2 truncation error tolerance for the
        local pod based on the number of input vectors `num_vecs` and the
        total number of snapshot vectors below the given node `snap_count`.
    product
        Inner product |Operator| w.r.t. which to compute the POD.
    pod_method
        A function `pod_method(U, eps, root_node, product)` for computing
        the POD of the |VectorArray| `U` w.r.t. the given inner product
        `product` and the l2 error tolerance `eps`. `root_node` is set to
        `True` when the POD is computed at the root of the tree.
    executor
        If not `None`, a :class:`concurrent.futures.Executor` object to use
        for parallelization.
    eval_snapshots_in_executor
        If `True` also parallelize the evaluation of the snapshot map.

    Returns
    -------
    modes
        The computed POD modes.
    svals
        The associated singular values.
    snap_count
        The total number of input snapshot vectors.
    """
    logger = getLogger('pymor.algorithms.hapod.hapod')

    async def hapod_step(node):
        children = tree.children(node)
        if children:
            modes, svals, snap_counts = zip(*await asyncio.gather(*(hapod_step(c) for c in children)))
            for m, sv in zip(modes, svals):
                m.scal(sv)
            U = modes[0]
            for V in modes[1:]:
                U.append(V, remove_from_other=True)
            snap_count = sum(snap_counts)
        else:
            if eval_snapshots_in_executor:
                U = await executor.submit(snapshots, node)
            else:
                U = snapshots(node)
            snap_count = len(U)

        with logger.block(f'Processing node {node}'):
            eps = local_eps(node, snap_count, len(U))
            if eps:
                modes, svals = await executor.submit(pod_method, U, eps, node == tree.root, product)
                return modes, svals, snap_count
            else:
                return U.copy(), np.ones(len(U)), snap_count

    # setup asyncio event loop
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        # probably we have closed the event loop ourselves in an earlier hapod call
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop_was_running = loop.is_running()

    # wrap Executer to ensure LIFO ordering of tasks
    # this ensures that PODs of parent nodes are computed as soon as all input data
    # is available
    if executor is not None:
        executor = LifoExecutor(executor)
    else:
        executor = FakeExecutor

    # perform HAPOD
    result = loop.run_until_complete(hapod_step(tree.root))

    # shutdown event loop
    if not loop_was_running:  # we haven't been inside a running event loop
        loop.close()

    return result


def inc_hapod(steps, snapshots, eps, omega, product=None, executor=None, eval_snapshots_in_executor=False):
    """Incremental Hierarchical Approximate POD.

    This computes the incremental HAPOD from :cite:`HLR18`.

    Parameters
    ----------
    steps
        The number of incremental POD updates.
    snapshots
        A mapping `snapshots(step)` returning for each incremental POD
        step the associated snapshot vectors.
    eps
        Desired l2-mean approximation error.
    omega
        Tuning parameter (0 < omega < 1) to balance performance with
        approximation quality.
    product
        Inner product |Operator| w.r.t. which to compute the POD.
    executor
        If not `None`, a :class:`concurrent.futures.Executor` object to use
        for parallelization.
    eval_snapshots_in_executor
        If `True` also parallelize the evaluation of the snapshot map.

    Returns
    -------
    modes
        The computed POD modes.
    svals
        The associated singular values.
    snap_count
        The total number of input snapshot vectors.
    """
    tree = IncHAPODTree(steps)
    return hapod(tree,
                 lambda node: snapshots(-node - 1),
                 std_local_eps(tree, eps, omega, False),
                 product=product,
                 executor=executor,
                 eval_snapshots_in_executor=eval_snapshots_in_executor)


def dist_hapod(num_slices, snapshots, eps, omega, product=None, executor=None, eval_snapshots_in_executor=False):
    """Distributed Hierarchical Approximate POD.

    This computes the distributed HAPOD from :cite:`HLR18`.

    Parameters
    ----------
    num_slices
        The number of snapshot vector slices.
    snapshots
        A mapping `snapshots(slice)` returning for each slice number
        the associated snapshot vectors.
    eps
        Desired l2-mean approximation error.
    omega
        Tuning parameter (0 < omega < 1) to balance performance with
        approximation quality.
    product
        Inner product |Operator| w.r.t. which to compute the POD.
    executor
        If not `None`, a :class:`concurrent.futures.Executor` object to use
        for parallelization.
    eval_snapshots_in_executor
        If `True` also parallelize the evaluation of the snapshot map.

    Returns
    -------
    modes
        The computed POD modes.
    svals
        The associated singular values.
    snap_count
        The total number of input snapshot vectors.
    """
    tree = DistHAPODTree(num_slices)
    return hapod(tree,
                 snapshots,
                 std_local_eps(tree, eps, omega, True),
                 product=product, executor=executor,
                 eval_snapshots_in_executor=eval_snapshots_in_executor)


def inc_vectorarray_hapod(steps, U, eps, omega, product=None, executor=None):
    """Incremental Hierarchical Approximate POD.

    This computes the incremental HAPOD from :cite:`HLR18` for a given |VectorArray|.

    Parameters
    ----------
    steps
        The number of incremental POD updates.
    U
        The |VectorArray| of which to compute the HAPOD.
    eps
        Desired l2-mean approximation error.
    omega
        Tuning parameter (0 < omega < 1) to balance performance with
        approximation quality.
    product
        Inner product |Operator| w.r.t. which to compute the POD.
    executor
        If not `None`, a :class:`concurrent.futures.Executor` object to use
        for parallelization.
    eval_snapshots_in_executor
        If `True` also parallelize the evaluation of the snapshot map.

    Returns
    -------
    modes
        The computed POD modes.
    svals
        The associated singular values.
    snap_count
        The total number of input snapshot vectors.
    """
    chunk_size = ceil(len(U) / steps)
    slices = range(0, len(U), chunk_size)
    return inc_hapod(len(slices),
                     lambda i: U[slices[i]: slices[i]+chunk_size],
                     eps, omega, product=product, executor=executor)


def inc_model_hapod(m, mus, num_steps_per_chunk, eps, omega, product=None):
    """Incremental Hierarchical Approximate POD.

    This computes the incremental HAPOD from [HLR18]_ for a given instationary |Model| and given parameters during
    timestepping.

    Note: currently restricted to time steppers with either num_values or nt specified (i.e. those with an apriori
          fixed trajectory length).

    Parameters
    ----------
    m
        The model used to obtain the solution trajectories.
	mus
		List of parameters used to obtain solution trajectories.
    num_steps_per_chunk
        The maximum number of vectors to consider for a single POD.
    eps
        Desired l2-mean approximation error.
    omega
        Tuning parameter (0 < omega < 1) to balance performance with
        approximation quality.
    product
        Inner product |Operator| w.r.t. which to compute the POD.

    Returns
    -------
    modes
        The computed POD modes.
    svals
        The associated singular values.
    snap_count
        The total number of input snapshot vectors.
    """
    logger = getLogger('pymor.algorithms.hapod.inc_model_hapod')

    assert isinstance(mus, (tuple, list))
    num_trajectories = len(mus)

    assert m.time_stepper.num_values or m.time_stepper.nt #
    num_steps_per_trajectory = m.time_stepper.num_values or m.time_stepper.nt

    lock = Lock() # does not matter which kind of lock we use
    persistent_data = {
        'mu_ind': 0,
        't': m.time_stepper.initial_time,
        'data': None}

    # function to be called within hapod
    def compute_next_snapshots(step):
        if lock.locked():
            raise RuntimeError('Not implemented for parallel executors yet!')
        with lock:
            U = m.solution_space.empty(reserve=num_steps_per_chunk)
            if persistent_data['mu_ind'] >= num_trajectories:
                logger.debug('all mus processed, returning empty U')
                return U
            while persistent_data['mu_ind'] < num_trajectories:
                mu = mus[persistent_data['mu_ind']]
                if not persistent_data['data']:
                    logger.debug(f'bootstrapping for mu={mu} ...')
                    # we are the first to process this mu, prepare
                    persistent_data['t'], _, persistent_data['data'] = \
                        m._compute_solution_bootstrap(mu=mu)
                # get data for this mu
                t, data = persistent_data['t'], persistent_data['data']
                # compute steps
                if not (t > m.T or np.allclose(t, m.T)):
                    logger.debug(f'stepping for mu={mu} ...')
                while not (t > m.T or np.allclose(t, m.T)):
                    t, U_t = m._compute_solution_step(t=t, data=data, mu=mu)
                    U.append(U_t)
                    if len(U) == num_steps_per_chunk:
                        logger.debug(f'  reached maximum chunk length of {num_steps_per_chunk} for mu={mu}, interrupting!')
                        # we are done with this U, save checkpoint and exit
                        persistent_data['t'] = t
                        persistent_data['data'] = data
                        return U
                # we are done with this trajectory, but U is not full
                # reset persistent data and continue with the next mu
                logger.debug(f'  done stepping for mu={mu}!')
                persistent_data['t'] = m.time_stepper.initial_time
                persistent_data['data'] = None
                persistent_data['mu_ind'] += 1
                continue
            return U

    logger.info(f'computing HAPOD of {num_trajectories} trajectories of length {num_steps_per_trajectory} each ...')

    return inc_hapod(steps=int(np.ceil((num_trajectories*num_steps_per_trajectory)/num_steps_per_chunk + 1)),
                     snapshots=compute_next_snapshots,
                     eps=eps,
                     omega=omega,
                     product=product,
                     executor=None,
                     eval_snapshots_in_executor=False)


def dist_vectorarray_hapod(num_slices, U, eps, omega, product=None, executor=None):
    """Distributed Hierarchical Approximate POD.

    This computes the distributed HAPOD from :cite:`HLR18` of a given |VectorArray|.

    Parameters
    ----------
    num_slices
        The number of snapshot vector slices.
    U
        The |VectorArray| of which to compute the HAPOD.
    eps
        Desired l2-mean approximation error.
    omega
        Tuning parameter (0 < omega < 1) to balance performance with
        approximation quality.
    product
        Inner product |Operator| w.r.t. which to compute the POD.
    executor
        If not `None`, a :class:`concurrent.futures.Executor` object to use
        for parallelization.

    Returns
    -------
    modes
        The computed POD modes.
    svals
        The associated singular values.
    snap_count
        The total number of input snapshot vectors.
    """
    chunk_size = ceil(len(U) / num_slices)
    slices = range(0, len(U), chunk_size)
    return dist_hapod(len(slices),
                      lambda i: U[slices[i]: slices[i]+chunk_size],
                      eps, omega, product=product, executor=executor)


def std_local_eps(tree, eps, omega, pod_on_leafs=True):

    L = tree.depth if pod_on_leafs else tree.depth - 1

    def local_eps(node, snap_count, input_count):
        if node == tree.root:
            return np.sqrt(snap_count) * omega * eps
        elif not pod_on_leafs and tree.is_leaf(node):
            return 0.
        else:
            return np.sqrt(snap_count) / np.sqrt(L - 1) * np.sqrt(1 - omega**2) * eps

    return local_eps


class LifoExecutor:

    def __init__(self, executor, max_workers=None):
        self.executor = executor
        self.max_workers = max_workers or executor._max_workers
        self.queue = LifoQueue()
        self.loop = asyncio.get_event_loop()
        self.sem = asyncio.Semaphore(self.max_workers)

    def submit(self, f, *args):
        future = self.loop.create_future()
        self.queue.put((future, f, args))
        self.loop.create_task(self.run_task())
        return future

    async def run_task(self):
        await self.sem.acquire()
        future, f, args = self.queue.get()
        executor_future = self.loop.run_in_executor(self.executor, f, *args)
        executor_future.add_done_callback(lambda f, ff=future: self.done_callback(future, f))

    def done_callback(self, future, executor_future):
        self.sem.release()
        future.set_result(executor_future.result())


class FakeExecutor:

    @staticmethod
    async def submit(f, *args):
        return f(*args)
