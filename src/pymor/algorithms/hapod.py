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
                modes, svals = pod(U, atol=0., rtol=0., l2_err=eps, product=product,
                                   orthonormalize=(node == tree.root), check=False)
                return modes, svals, snap_count
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
    return inc_hapod(len(slices),
                     lambda i: U[slices[i]: slices[i]+chunk_size],
                     eps, omega, product=product)


def dist_vectorarray_hapod(num_slices, U, eps, omega, product=None):
    chunk_size = ceil(len(U) / num_slices)
    slices = range(0, len(U), chunk_size)
    return dist_hapod(len(slices),
                      lambda i: U[slices[i]: slices[i]+chunk_size],
                      eps, omega, product=product)


if __name__ == '__main__':
    from time import time

    from pymor.basic import *
    from pymor.algorithms.hapod import *
    from pymor.tools.table import format_table

    p = burgers_problem_2d()
    d, _ = discretize_instationary_fv(p, nt=400)

    U = d.solution_space.empty()
    for mu in d.parameter_space.sample_randomly(5):
        U.append(d.solve(mu))

    tic = time()
    pod_modes = pod(U, l2_err=1e-2 * np.sqrt(len(U)), product=d.l2_product, check=False)[0]
    pod_time = time() - tic

    tic = time()
    dist_modes = dist_vectorarray_hapod(10, U, 1e-2, 0.75, product=d.l2_product)[0]
    dist_time = time() - tic

    tic = time()
    inc_modes = inc_vectorarray_hapod(100, U, 1e-2, 0.75, product=d.l2_product)[0]
    inc_time = time() - tic

    print('Snapshot matrix: {} x {}'.format(U.dim, len(U)))
    print(format_table([
        ['Method', 'Error', 'Modes', 'Time'],
        ['POD', np.linalg.norm(d.l2_norm(U-pod_modes.lincomb(d.l2_product.apply2(U, pod_modes)))/np.sqrt(len(U))),
         len(pod_modes), pod_time],
        ['DIST HAPOD', np.linalg.norm(d.l2_norm(U-dist_modes.lincomb(d.l2_product.apply2(U, dist_modes)))/np.sqrt(len(U))),
         len(dist_modes), dist_time],
        ['INC HAPOD', np.linalg.norm(d.l2_norm(U-inc_modes.lincomb(d.l2_product.apply2(U, inc_modes)))/np.sqrt(len(U))),
         len(inc_modes), inc_time]]
    ))
