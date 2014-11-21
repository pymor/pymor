# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip
from numbers import Number
import numpy as np

from pymor.la.interfaces import VectorArrayInterface, VectorSpace


class BlockVectorArray(VectorArrayInterface):
    """|VectorArray| implementation
    """

    def __init__(self, blocks, block_sizes=None, copy=False):
        if isinstance(blocks, list):
            # we assume we get a list of compatible vector arrays
            assert all([isinstance(block, VectorArrayInterface) for block in blocks])
            # to begin with we keep the given block sizes
            assert block_sizes is None
            assert len(blocks) > 0
            assert all([isinstance(block, VectorArrayInterface) for block in blocks])
            self._blocks = [block.copy() for block in blocks] if copy else blocks
        else:
            # we assume we are given a vector array and a list of block sizes
            # we slice the vector into appropriate blocks and create vector arrays
            assert isinstance(blocks, VectorArrayInterface)
            assert block_sizes is not None
            assert isinstance(block_sizes, list)
            assert all(isinstance(block_size, Number) for block_size in block_sizes)
            assert blocks.dim == sum(block_sizes)
            block_type = blocks.space.type
            self._blocks = [block_type(blocks.components(range(sum(block_sizes[:ss]), sum(block_sizes[:(ss + 1)]))))
                            for ss in np.arange(len(block_sizes))]
        assert self._blocks_are_valid()

    def _blocks_are_valid(self):
        return all([len(block) == len(self._blocks[0]) for block in self._blocks])

    @classmethod
    def make_array(cls, subtype=None, count=0, reserve=0):
        assert isinstance(subtype, list)
        assert all([isinstance(subspace, VectorSpace) for subspace in subtype])
        return BlockVectorArray([subspace.type.make_array(subspace.subtype,
                                                          count=count,
                                                          reserve=reserve) for subspace in subtype])

    def block(self, ind):
        """
        Returns a copy of each block (no slicing).
        """
        assert self._blocks_are_valid()
        if isinstance(ind, list):
            assert all(isinstance(ii, Number) for ii in ind)
            return [self._blocks[ii].copy() for ii in ind]
        else:
            assert isinstance(ind, Number)
            return self._blocks[ind].copy()

    @property
    def num_blocks(self):
        return len(self.subtype)

    @property
    def subtype(self):
        return [block.space for block in self._blocks]

    def __len__(self):
        assert self._blocks_are_valid()
        return len(self._blocks[0])

    @property
    def dim(self):
        return sum([block.dim for block in self._blocks])

    def copy(self, ind=None):
        return BlockVectorArray([block.copy(ind) for block in self._blocks], copy=False)

    def append(self, other, o_ind=None, remove_from_other=False):
        assert self._blocks_are_valid()
        assert other in self.space
        for block, other_block in izip(self._blocks, other._blocks):
            block.append(other_block, o_ind=o_ind, remove_from_other=remove_from_other)

    def remove(self, ind):
        assert self.check_ind(ind)
        for block in self._blocks:
            block.remove(ind)

    def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
        raise NotImplementedError

    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        assert other in self.space
        return np.all([block.almost_equal(other_block, ind=ind, o_ind=o_ind, rtol=rtol, atol=atol)
                       for block, other_block in izip(self._blocks, other._blocks)])

    def scal(self, alpha, ind=None):
        for block in self._blocks:
            block.scal(alpha, ind=ind)

    def axpy(self, alpha, x, ind=None, x_ind=None):
        assert x in self.space
        if len(x) > 0:
            for block, x_block in izip(self._blocks, x._blocks):
                block.axpy(alpha, x_block, ind, x_ind)

    def dot(self, other, pairwise, ind=None, o_ind=None):
        assert other in self.space
        dots = [block.dot(other_block, pairwise, ind=ind, o_ind=o_ind)
                for block, other_block in izip(self._blocks, other._blocks)]
        assert all([dot.shape == dots[0].shape for dot in dots])
        ret = np.zeros(dots[0].shape)
        for dot in dots:
            ret += dot
        return ret

    def lincomb(self, coefficients, ind=None):
        raise NotImplementedError

    def l1_norm(self, ind=None):
        raise NotImplementedError

    def l2_norm(self, ind=None):
        assert self.check_ind(ind)
        if len(self) != 1:
            raise NotImplementedError
        assert (ind is None
                or (ind == 0 if isinstance(ind, Number)
                             else (len(ind) == 0 or (len(ind) == 1 and ind[0] == 0))))
        return np.sqrt(sum([block.l2_norm() for block in self._blocks]))

    def sup_norm(self, ind=None):
        raise NotImplementedError

    def components(self, component_indices, ind=None):
        raise NotImplementedError

    def amax(self, ind=None):
        raise NotImplementedError

    def gramian(self, ind=None):
        raise NotImplementedError

