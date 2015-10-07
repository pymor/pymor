# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip
from numbers import Number
import numpy as np

from pymor.vectorarrays.interfaces import VectorArrayInterface, VectorSpace


class BlockVectorArray(VectorArrayInterface):
    """|VectorArray| implementation
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

    def __len__(self):
        return len(self._blocks[0])

    @property
    def dim(self):
        return np.sum(self._dims)

    @property
    def data(self):
        return np.hstack([block.data for block in self.blocks])

    def copy(self, ind=None):
        assert self.check_ind(ind)
        return BlockVectorArray([block.copy(ind=ind) for block in self._blocks], copy=False)

    def append(self, other, o_ind=None, remove_from_other=False):
        assert self._blocks_are_valid()
        assert other in self.space
        for block, other_block in izip(self._blocks, other._blocks):
            block.append(other_block, o_ind=o_ind, remove_from_other=remove_from_other)

    def remove(self, ind=None):
        assert self.check_ind(ind)
        for block in self._blocks:
            block.remove(ind)

    def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
        assert other in self.space
        assert self.check_ind(ind)
        assert other.check_ind(o_ind)
        for block, o_block in zip(self._blocks, other._blocks):
            block.replace(o_block, ind=ind, o_ind=o_ind, remove_from_other=remove_from_other)

    def scal(self, alpha, ind=None):
        for block in self._blocks:
            block.scal(alpha, ind=ind)

    def axpy(self, alpha, x, ind=None, x_ind=None):
        assert x in self.space
        assert isinstance(alpha, Number) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (self.len_ind(ind),)
        if x.len_ind(x_ind) > 0:
            for block, x_block in izip(self._blocks, x._blocks):
                block.axpy(alpha, x_block, ind, x_ind)
        else:
            assert self.len_ind(ind) == 0

    def dot(self, other, ind=None, o_ind=None):
        assert other in self.space
        dots = [block.dot(other_block, ind=ind, o_ind=o_ind)
                for block, other_block in izip(self._blocks, other._blocks)]
        assert all([dot.shape == dots[0].shape for dot in dots])
        ret = np.zeros(dots[0].shape)
        for dot in dots:
            ret += dot
        return ret

    def pairwise_dot(self, other, ind=None, o_ind=None):
        assert other in self.space
        dots = [block.pairwise_dot(other_block, ind=ind, o_ind=o_ind)
                for block, other_block in izip(self._blocks, other._blocks)]
        assert all([dot.shape == dots[0].shape for dot in dots])
        ret = np.zeros(dots[0].shape)
        for dot in dots:
            ret += dot
        return ret

    def lincomb(self, coefficients, ind=None):
        assert self.check_ind(ind)
        lincombs = [block.lincomb(coefficients, ind=ind) for block in self._blocks]
        return BlockVectorArray(lincombs)

    def l1_norm(self, ind=None):
        assert self.check_ind(ind)
        return np.sum(np.array([block.l1_norm(ind=ind) for block in self._blocks]), axis=0)

    def l2_norm(self, ind=None):
        assert self.check_ind(ind)
        return np.sqrt(np.sum(np.array([block.l2_norm(ind=ind)**2 for block in self._blocks]),
                              axis=0))

    def sup_norm(self, ind=None):
        assert self.check_ind(ind)
        return np.max(np.array([block.sup_norm(ind=ind) for block in self._blocks]),
                      axis=0)

    def components(self, component_indices, ind=None):
        assert self.check_ind(ind)
        component_indices = np.array(component_indices)
        if not len(component_indices):
            return np.zeros((self.len_ind(ind), 0))

        bins = self._ind_bins
        block_inds = np.digitize(component_indices, bins) - 1
        component_indices -= bins[block_inds]
        blocks = self._nonempty_blocks
        return np.array([blocks[bi].components([ci], ind=ind)[:, 0] for bi, ci in zip(block_inds, component_indices)]).T

    def amax(self, ind=None):
        assert self.check_ind(ind)
        inds, vals = zip(*(block.amax(ind=ind) for block in self._nonempty_blocks))
        inds, vals = np.array(inds), np.array(vals)
        inds += self._ind_bins[:-1][..., np.newaxis]
        block_inds = np.argmax(vals, axis=0)
        ar = np.arange(inds.shape[1])
        return inds[block_inds, ar], vals[block_inds, ar]
