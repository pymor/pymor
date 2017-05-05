# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number
import numpy as np

from pymor.core.interfaces import classinstancemethod
from pymor.vectorarrays.interfaces import VectorArrayInterface, VectorSpaceInterface, _INDEXTYPES


class BlockVectorArray(VectorArrayInterface):
    """|VectorArray| where each vector is a direct sum of sub-vectors.

    Given a list of equal length |VectorArrays| `blocks`, this |VectorArray|
    represents the direct sums of the vectors contained in the arrays.
    The associated |VectorSpace| is :class:`BlockVectorSpace`.

    :class:`BlockVectorArray` can be used in conjunction with
    :class:`~pymor.operators.block.BlockOperator`.
    """

    def __init__(self, blocks, space):
        self._blocks = tuple(blocks)
        self.space = space
        assert self._blocks_are_valid()

    @property
    def data(self):
        return np.hstack([block.data for block in self._blocks])

    def block(self, ind):
        """
        Returns a copy of each block (no slicing).
        """
        assert self._blocks_are_valid()
        if isinstance(ind, (tuple, list)):
            assert all(isinstance(ii, Number) for ii in ind)
            return tuple(self._blocks[ii].copy() for ii in ind)
        else:
            assert isinstance(ind, Number)
            return self._blocks[ind].copy()

    @property
    def num_blocks(self):
        return len(self._blocks)

    def __len__(self):
        return len(self._blocks[0])

    def __getitem__(self, ind):
        return BlockVectorArrayView(self, ind)

    def __delitem__(self, ind):
        assert self.check_ind(ind)
        for block in self._blocks:
            del block[ind]

    def append(self, other, remove_from_other=False):
        assert self._blocks_are_valid()
        assert other in self.space
        for block, other_block in zip(self._blocks, other._blocks):
            block.append(other_block, remove_from_other=remove_from_other)

    def copy(self, deep=False):
        return BlockVectorArray([block.copy(deep) for block in self._blocks], self.space)

    def scal(self, alpha):
        for block in self._blocks:
            block.scal(alpha)

    def axpy(self, alpha, x):
        assert x in self.space
        assert isinstance(alpha, _INDEXTYPES) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (len(self),)
        if len(x) > 0:
            for block, x_block in zip(self._blocks, x._blocks):
                block.axpy(alpha, x_block)
        else:
            assert len(self) == 0

    def dot(self, other):
        assert other in self.space
        dots = [block.dot(other_block) for block, other_block in zip(self._blocks, other._blocks)]
        assert all([dot.shape == dots[0].shape for dot in dots])
        ret = np.zeros(dots[0].shape)
        for dot in dots:
            ret = ret + dot
        return ret

    def pairwise_dot(self, other):
        assert other in self.space
        dots = [block.pairwise_dot(other_block)
                for block, other_block in zip(self._blocks, other._blocks)]
        assert all([dot.shape == dots[0].shape for dot in dots])
        ret = np.zeros(dots[0].shape)
        for dot in dots:
            ret = ret + dot
        return ret

    def lincomb(self, coefficients):
        lincombs = [block.lincomb(coefficients) for block in self._blocks]
        return BlockVectorArray(lincombs, self.space)

    def l1_norm(self):
        return np.sum(np.array([block.l1_norm() for block in self._blocks]), axis=0)

    def l2_norm(self):
        return np.sqrt(np.sum(np.array([block.l2_norm2() for block in self._blocks]), axis=0))

    def l2_norm2(self):
        return np.sum(np.array([block.l2_norm2() for block in self._blocks]), axis=0)

    def sup_norm(self):
        return np.max(np.array([block.sup_norm() for block in self._blocks]), axis=0)

    def components(self, component_indices):
        component_indices = np.array(component_indices)
        if not len(component_indices):
            return np.zeros((len(self), 0))

        self._compute_bins()
        block_inds = np.digitize(component_indices, self._bins) - 1
        component_indices -= self._bins[block_inds]
        block_inds = self._bin_map[block_inds]
        blocks = self._blocks
        return np.array([blocks[bi].components([ci])[:, 0]
                         for bi, ci in zip(block_inds, component_indices)]).T

    def amax(self):
        self._compute_bins()
        blocks = self._blocks
        inds, vals = zip(*(blocks[bi].amax() for bi in self._bin_map))
        inds, vals = np.array(inds), np.array(vals)
        inds += self._bins[:-1][..., np.newaxis]
        block_inds = np.argmax(vals, axis=0)
        ar = np.arange(inds.shape[1])
        return inds[block_inds, ar], vals[block_inds, ar]

    def _blocks_are_valid(self):
        return all([len(block) == len(self._blocks[0]) for block in self._blocks])

    def _compute_bins(self):
        if not hasattr(self, '_bins'):
            dims = np.array([subspace.dim for subspace in self.space.subspaces])
            self._bin_map = bin_map = np.where(dims > 0)[0]
            self._bins = np.cumsum(np.hstack(([0], dims[bin_map])))


class BlockVectorSpace(VectorSpaceInterface):
    """|VectorSpace| of :class:`BlockVectorArrays <BlockVectorArray>`.

    A :class:`BlockVectorSpace` is defined by the |VectorSpaces| of the
    individual subblocks which constitute a given array. In particular
    for a given :class`BlockVectorArray` `U`, we have the identity ::

        (U.blocks[0].space, U.blocks[1].space, ..., U.blocks[-1].space) == U.space.

    Parameters
    ----------
    subspaces
        The tuple defined above.
    """

    def __init__(self, subspaces, id_=None):
        subspaces = tuple(subspaces)
        assert all([isinstance(subspace, VectorSpaceInterface) for subspace in subspaces])
        self.subspaces = subspaces
        self.id = id_

    def __eq__(self, other):
        return (type(other) is BlockVectorSpace and
                len(self.subspaces) == len(other.subspaces) and
                all(space == other_space for space, other_space in zip(self.subspaces, other.subspaces)))

    def __hash__(self):
        return sum(hash(s) for s in self.subspaces) + hash(self.id)

    @property
    def dim(self):
        return sum(subspace.dim for subspace in self.subspaces)

    def zeros(self, count=1, reserve=0):
        return BlockVectorArray([subspace.zeros(count=count, reserve=reserve) for subspace in self.subspaces], self)

    @classinstancemethod
    def make_array(cls, obj, id_=None):
        assert len(obj) > 0
        return cls(tuple(o.space for o in obj), id_=id_).make_array(obj)

    @make_array.instancemethod
    def make_array(self, obj):
        assert len(obj) == len(self.subspaces)
        assert all(block in subspace for block, subspace in zip(obj, self.subspaces))
        return BlockVectorArray(obj, self)

    def from_data(self, data):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data_ind = np.cumsum([0] + [subspace.dim for subspace in self.subspaces])
        return BlockVectorArray([subspace.from_data(data[:, data_ind[i]:data_ind[i + 1]])
                                 for i, subspace in enumerate(self.subspaces)], self)


class BlockVectorArrayView(BlockVectorArray):

    is_view = True

    def __init__(self, base, ind):
        self._blocks = tuple(block[ind] for block in base._blocks)
        self.space = base.space
