# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number
import numpy as np

from pymor.vectorarrays.interfaces import VectorArrayInterface, VectorSpace, _INDEXTYPES


class BlockVectorArray(VectorArrayInterface):
    """|VectorArray| where each vector is a direct sum of sub-vectors.

    Given a list of equal length |VectorArrays| `blocks`, this |VectorArray|
    represents the direct sums of the vectors contained in the arrays.

    The :attr:`~pymor.vectorarrays.interfaces.VectorArrayInterface.subtype`
    of the array will be the tuple ::

        (blocks[0].space, blocks[1].space, ..., blocks[-1].space).

    :class:`BlockVectorArray` can be used in conjunction with
    :class:`~pymor.operators.block.BlockOperator`.


    Parameters
    ----------
    blocks
        The list of sub-arrays.
    copy
        If `True`, copy all arrays contained in `blocks`.
    """

    def __init__(self, blocks, copy=False):
        assert isinstance(blocks, (tuple, list))
        assert all([isinstance(block, VectorArrayInterface) for block in blocks])
        assert len(blocks) > 0
        self._blocks = tuple(block.copy() for block in blocks) if copy else tuple(blocks)
        self._nonempty_blocks = tuple(block for block in blocks if block.dim > 0)
        self._dims = np.array([block.dim for block in blocks])
        self._nonempty_dims = np.array([block.dim for block in self._nonempty_blocks])
        self._ind_bins = np.cumsum([0] + [block.dim for block in self._nonempty_blocks])
        assert self._blocks_are_valid()

    def _blocks_are_valid(self):
        return all([len(block) == len(self._blocks[0]) for block in self._blocks])

    @classmethod
    def make_array(cls, subtype=None, count=0, reserve=0):
        assert isinstance(subtype, tuple)
        assert all([isinstance(subspace, VectorSpace) for subspace in subtype])
        return BlockVectorArray([subspace.type.make_array(subspace.subtype,
                                                          count=count,
                                                          reserve=reserve) for subspace in subtype])

    @classmethod
    def from_data(cls, data, subtype):
        assert isinstance(subtype, tuple)
        assert all([isinstance(subspace, VectorSpace) for subspace in subtype])
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data_ind = np.cumsum([0] + [subspace.dim for subspace in subtype])
        return cls([subspace.type.from_data(data[:, data_ind[i]:data_ind[i + 1]], subspace.subtype)
                    for i, subspace in enumerate(subtype)])

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
        return len(self.subtype)

    @property
    def subtype(self):
        return tuple(block.space for block in self._blocks)

    @property
    def dim(self):
        return np.sum(self._dims)

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
        return BlockVectorArray([block.copy(deep) for block in self._blocks], copy=False)

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
            ret += dot
        return ret

    def pairwise_dot(self, other):
        assert other in self.space
        dots = [block.pairwise_dot(other_block)
                for block, other_block in zip(self._blocks, other._blocks)]
        assert all([dot.shape == dots[0].shape for dot in dots])
        ret = np.zeros(dots[0].shape)
        for dot in dots:
            ret += dot
        return ret

    def lincomb(self, coefficients):
        lincombs = [block.lincomb(coefficients) for block in self._blocks]
        return BlockVectorArray(lincombs)

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

        bins = self._ind_bins
        block_inds = np.digitize(component_indices, bins) - 1
        component_indices -= bins[block_inds]
        blocks = self._nonempty_blocks
        return np.array([blocks[bi].components([ci])[:, 0]
                         for bi, ci in zip(block_inds, component_indices)]).T

    def amax(self):
        inds, vals = zip(*(block.amax() for block in self._nonempty_blocks))
        inds, vals = np.array(inds), np.array(vals)
        inds += self._ind_bins[:-1][..., np.newaxis]
        block_inds = np.argmax(vals, axis=0)
        ar = np.arange(inds.shape[1])
        return inds[block_inds, ar], vals[block_inds, ar]


class BlockVectorArrayView(BlockVectorArray):

    is_view = True

    def __init__(self, base, ind):
        self._blocks = tuple(block[ind] for block in base._blocks)
        self._nonempty_blocks = tuple(block for block in self._blocks if block.dim > 0)
        self._dims = base._dims
        self._nonempty_dims = base._nonempty_dims
        self._ind_bins = base._ind_bins

    @property
    def space(self):
        return VectorSpace(BlockVectorArray, self.subtype)
