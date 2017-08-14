# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from math import ceil
import numpy as np

from pymor.algorithms.pod import pod
from pymor.core.interfaces import BasicInterface, abstractmethod
from pymor.core.logger import getLogger


class Tree(BasicInterface):

    root = 0

    @abstractmethod
    def children(self, node):
        pass

    @property
    def depth(self):
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


class DistHAPODTree(Tree):

    def __init__(self, slices):
        self.root = slices

    def children(self, node):
        return tuple(range(self.root)) if node == self.root else ()


def hapod(tree, snapshots, local_eps, product=None):

    logger = getLogger('pymor.algorithms.hapod.hapod')

    def hapod_step(node):
        children = tree.children(node)
        if children:
            modes, svals, snap_counts = zip(*(hapod_step(c) for c in children))
            for m, sv in zip(modes, svals):
                m.scal(sv)
            U = modes[0]
            for V in modes[1:]:
                U.append(V, remove_from_other=True)
            snap_count = sum(snap_counts)
        else:
            U = snapshots(node)
            snap_count = len(U)

        with logger.block('Processing node {}'.format(node)):
            eps = local_eps(node, snap_count, len(U))
            if eps:
                return (*pod(U, atol=0., rtol=0., l2_err=eps, product=product,
                             orthonormalize=(node == tree.root), check=False),
                        snap_count)
            else:
                return U.copy(), np.ones(len(U)), snap_count

    return hapod_step(tree.root)


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


def inc_hapod(steps, snapshots, eps, omega, product=None):
    tree = IncHAPODTree(steps)
    return hapod(tree,
                 lambda node: snapshots(-node - 1),
                 std_local_eps(tree, eps, omega, False),
                 product=product)


def dist_hapod(num_slices, snapshots, eps, omega, product=None):
    tree = DistHAPODTree(num_slices)
    return hapod(tree,
                 snapshots,
                 std_local_eps(tree, eps, omega, True),
                 product=product)


def inc_vectorarray_hapod(steps, U, eps, omega, product=None):
    chunk_size = ceil(len(U) / steps)
    slices = range(0, len(U), chunk_size)
    return inc_hapod(steps,
                     lambda i: U[slices[i]: slices[i]+chunk_size],
                     eps, omega, product=product)


def dist_vectorarray_hapod(num_slices, U, eps, omega, product=None):
    chunk_size = ceil(len(U) / num_slices)
    slices = range(0, len(U), chunk_size)
    return dist_hapod(num_slices,
                      lambda i: U[slices[i]: slices[i]+chunk_size],
                      eps, omega, product=product)
