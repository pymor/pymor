# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import asyncio
from collections import defaultdict
from math import ceil
import numpy as np
from threading import Thread

from pymor.algorithms.pod import pod
from pymor.core.logger import getLogger


class Node:

    def __init__(self, tag=None, parent=None, after=None):
        after = after or []
        self.tag, self.parent, self.after = tag, parent, after
        self.children = []
        if parent:
            parent.children.append(self)

    def add_child(self, tag=None, after=None, **kwargs):
        return Node(tag=tag, parent=self, after=after, **kwargs)

    @property
    def depth(self):
        max_level = 0
        for _, level in self.traverse(True):
            max_level = max(max_level, level)
        return max_level + 1

    @property
    def is_leaf(self):
        return not self.children

    @property
    def is_root(self):
        return not self.parent

    def traverse(self, return_level=False):
        current_node = self
        last_child = None
        level = 0
        while True:
            if last_child is None:
                if return_level:
                    yield current_node, level
                else:
                    yield current_node
            if current_node.children:
                if last_child is None:
                    current_node = current_node.children[0]
                    level += 1
                    continue
                else:
                    last_child_pos = current_node.children.index(last_child)
                    if last_child_pos + 1 < len(current_node.children):
                        current_node = current_node.children[last_child_pos+1]
                        last_child = None
                        level += 1
                        continue
            if not current_node.parent:
                return
            last_child = current_node
            current_node = current_node.parent
            level -= 1

    def __str__(self):
        lines = []
        for node, level in self.traverse(True):
            line = ''
            if node.parent:
                p = node.parent
                while p.parent:
                    if p.parent.children.index(p) + 1 == len(p.parent.children):
                        line = '  ' + line
                    else:
                        line = '| ' + line
                    p = p.parent
                line += '+-o'
            else:
                line += 'o'
            if node.tag is not None:
                line += f' {node.tag}'
            if node.after:
                line += f' (after {",".join(str(a) for a in node.after)})'
            lines.append(line)
        return '\n'.join(lines)


def inc_hapod_tree(steps):
    tree = node = Node()
    for step in range(steps)[::-1]:
        # add leaf node for a new snapshot
        node.add_child(tag=step, after=(step-1,) if step > 0 else None)
        if step > 0:
            # add node for the previous POD step
            node = node.add_child()
    return tree


def dist_hapod_tree(num_slices, arity=None):
    tree = Node()
    if arity is None:
        arity = num_slices

    def add_children(node, slices):
        if len(slices) > arity:
            sub_slices = np.array_split(slices, arity)
            for s in sub_slices:
                if len(s) > 1:
                    child = node.add_child()
                    add_children(child, s)
                else:
                    child = node.add_child(s.item())
        else:
            for s in slices:
                node.add_child(s)

    add_children(tree, np.arange(num_slices))

    return tree


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

    node_finished_events = defaultdict(asyncio.Event)

    async def hapod_step(node):
        if node.after:
            await asyncio.wait([asyncio.create_task(node_finished_events[a].wait()) for a in node.after])

        if node.children:
            modes, svals, snap_counts = zip(
                *await asyncio.gather(*(hapod_step(c) for c in node.children))
            )
            for m, sv in zip(modes, svals):
                m.scal(sv)
            U = modes[0]
            for V in modes[1:]:
                U.append(V, remove_from_other=True)
            snap_count = sum(snap_counts)
        else:
            logger.info(f'Obtaining snapshots for node {node.tag or ""} ...')
            if eval_snapshots_in_executor:
                U = await executor.submit(snapshots, node)
            else:
                U = snapshots(node)
            snap_count = len(U)

        eps = local_eps(node, snap_count, len(U))
        if eps:
            logger.info('Computing intermediate POD ...')
            modes, svals = await executor.submit(pod_method, U, eps, not node.parent, product)
        else:
            modes, svals = U.copy(), np.ones(len(U))
        if node.tag is not None:
            node_finished_events[node.tag].set()
        return modes, svals, snap_count

    # wrap Executer to ensure LIFO ordering of tasks
    # this ensures that PODs of parent nodes are computed as soon as all input data
    # is available
    if executor is not None:
        executor = LifoExecutor(executor)
    else:
        executor = FakeExecutor

    # run new asyncio event loop in separate thread to not interfere with
    # already running event loops (e.g. jupyter)

    def main():
        nonlocal result
        result = asyncio.run(hapod_step(tree))
    result = None
    hapod_thread = Thread(target=main)
    hapod_thread.start()
    hapod_thread.join()
    return result


def inc_hapod(steps, snapshots, eps, omega, product=None, executor=None):
    """Incremental Hierarchical Approximate POD.

    This computes the incremental HAPOD from :cite:`HLR18`.

    Parameters
    ----------
    steps
        The number of incremental POD updates. Has to agree with the length
        of `snapshots`.
    snapshots
        An iterable returning for each incremental POD step the associated
        snapshot vectors.
    eps
        Desired l2-mean approximation error.
    omega
        Tuning parameter (0 < omega < 1) to balance performance with
        approximation quality.
    product
        Inner product |Operator| w.r.t. which to compute the POD.
    executor
        If not `None`, a :class:`concurrent.futures.Executor` object to use
        to compute new snapshot vectors and POD updates in parallel.

    Returns
    -------
    modes
        The computed POD modes.
    svals
        The associated singular values.
    snap_count
        The total number of input snapshot vectors.
    """
    tree = inc_hapod_tree(steps)

    last_step = -1
    snapshots = iter(snapshots)

    def get_snapshots(node):
        nonlocal last_step
        assert node.tag == last_step + 1
        last_step += 1
        return next(snapshots)

    result = hapod(tree,
                   get_snapshots,
                   std_local_eps(tree, eps, omega, False),
                   product=product,
                   executor=executor,
                   eval_snapshots_in_executor=True)
    assert last_step == steps - 1
    return result


def dist_hapod(num_slices, snapshots, eps, omega, arity=None,
               product=None, executor=None, eval_snapshots_in_executor=False):
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
    arity
        If not `None`, the arity of the HAPOD tree. Otherwise, a
        tree of depth 2 is used (one POD per slice and one additional
        POD of the resulting data).
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
    tree = dist_hapod_tree(num_slices, arity)
    return hapod(tree,
                 snapshots,
                 std_local_eps(tree, eps, omega, True),
                 product=product, executor=executor,
                 eval_snapshots_in_executor=eval_snapshots_in_executor)


def inc_vectorarray_hapod(steps, U, eps, omega, product=None):
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

    def snapshots():
        for slice in slices:
            yield U[slice: slice+chunk_size]

    return inc_hapod(len(slices), snapshots(),
                     eps, omega, product=product)


def dist_vectorarray_hapod(num_slices, U, eps, omega, arity=None, product=None, executor=None):
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
    arity
        If not `None`, the arity of the HAPOD tree. Otherwise, a
        tree of depth 2 is used (one POD per slice and one additional
        POD of the resulting data).
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
                      lambda i: U[slices[i.tag]: slices[i.tag]+chunk_size],
                      eps, omega, arity=arity, product=product, executor=executor)


def std_local_eps(tree, eps, omega, pod_on_leafs=True):

    L = tree.depth if pod_on_leafs else tree.depth - 1

    def local_eps(node, snap_count, input_count):
        if node.is_root:
            return np.sqrt(snap_count) * omega * eps
        elif node.is_leaf and not pod_on_leafs:
            return 0.
        else:
            return np.sqrt(snap_count) / np.sqrt(L - 1) * np.sqrt(1 - omega**2) * eps

    return local_eps


class LifoExecutor:

    def __init__(self, executor, max_workers=None):
        self.executor = executor
        self.max_workers = max_workers or executor._max_workers
        self.queue = []

    def submit(self, f, *args):
        future = asyncio.get_event_loop().create_future()
        self.queue.append((future, f, args))
        asyncio.get_event_loop().create_task(self.run_task())
        return future

    async def run_task(self):
        if not hasattr(self, 'sem'):
            self.sem = asyncio.Semaphore(self.max_workers)
        await self.sem.acquire()
        future, f, args = self.queue.pop()
        executor_future = asyncio.get_event_loop().run_in_executor(self.executor, f, *args)
        executor_future.add_done_callback(lambda f, ff=future: self.done_callback(future, f))

    def done_callback(self, future, executor_future):
        self.sem.release()
        future.set_result(executor_future.result())


class FakeExecutor:

    @staticmethod
    async def submit(f, *args):
        return f(*args)
