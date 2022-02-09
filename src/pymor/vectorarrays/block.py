# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from functools import reduce
from numbers import Number
import numpy as np

from pymor.core.base import classinstancemethod
from pymor.vectorarrays.interface import VectorArray, VectorArrayImpl, VectorSpace


class BlockVectorArrayImpl(VectorArrayImpl):

    def __init__(self, blocks, space):
        self._blocks = tuple(blocks)
        self.space = space
        assert self._blocks_are_valid()

    def __len__(self):
        try:
            return len(self._blocks[0])
        except IndexError:
            return 0

    def to_numpy(self, ensure_copy, ind):
        if len(self._blocks):
            # hstack will error out with empty input list
            return np.hstack([_indexed(block, ind).to_numpy(False) for block in self._blocks])
        else:
            return np.empty((0, 0))

    def real(self, ind):
        return type(self)([_indexed(block, ind).real for block in self._blocks], self.space)

    def imag(self, ind):
        return type(self)([_indexed(block, ind).imag for block in self._blocks], self.space)

    def conj(self, ind):
        return type(self)([_indexed(block, ind).conj() for block in self._blocks], self.space)

    def delete(self, ind):
        ind = slice(None) if ind is None else ind
        for block in self._blocks:
            del block[ind]

    def append(self, other, remove_from_other, oind):
        assert self._blocks_are_valid()
        for block, other_block in zip(self._blocks, other._blocks):
            block.append(_indexed(other_block, oind), remove_from_other)

    def copy(self, deep, ind):
        return type(self)([_indexed(block, ind).copy(deep) for block in self._blocks], self.space)

    def scal(self, alpha, ind):
        for block in self._blocks:
            _indexed(block, ind).scal(alpha)

    def scal_copy(self, alpha, ind):
        if isinstance(alpha, Number):
            if alpha == -1:
                return type(self)([-_indexed(block, ind) for block in self._blocks], self.space)
        return super().scal_copy(alpha, ind)

    def axpy(self, alpha, x, ind, xind):
        for block, x_block in zip(self._blocks, x._blocks):
            _indexed(block, ind).axpy(alpha, _indexed(x_block, xind))

    def axpy_copy(self, alpha, x, ind, xind):
        if isinstance(alpha, Number):
            if alpha == 1:
                return type(self)([_indexed(block, ind) + _indexed(x_block, xind)
                                   for block, x_block in zip(self._blocks, x._blocks)],
                                  self.space)
            elif alpha == -1:
                return type(self)([_indexed(block, ind) - _indexed(x_block, xind)
                                   for block, x_block in zip(self._blocks, x._blocks)],
                                  self.space)
        return super().axpy_copy(alpha, x, ind, xind)

    def inner(self, other, ind, oind):
        prods = [_indexed(block, ind).inner(_indexed(other_block, oind))
                 for block, other_block in zip(self._blocks, other._blocks)]
        assert all([prod.shape == prods[0].shape for prod in prods])
        common_dtype = reduce(np.promote_types, (prod.dtype for prod in prods))
        ret = np.zeros(prods[0].shape, dtype=common_dtype)
        for prod in prods:
            ret += prod
        return ret

    def pairwise_inner(self, other, ind, oind):
        prods = [_indexed(block, ind).pairwise_inner(_indexed(other_block, oind))
                 for block, other_block in zip(self._blocks, other._blocks)]
        assert all([prod.shape == prods[0].shape for prod in prods])
        common_dtype = reduce(np.promote_types, (prod.dtype for prod in prods))
        ret = np.zeros(prods[0].shape, dtype=common_dtype)
        for prod in prods:
            ret += prod
        return ret

    def lincomb(self, coefficients, ind):
        lincombs = [_indexed(block, ind).lincomb(coefficients) for block in self._blocks]
        return type(self)(lincombs, self.space)

    def norm2(self, ind):
        return np.sum(np.array([_indexed(block, ind).norm2() for block in self._blocks]), axis=0)

    def dofs(self, dof_indices, ind):
        if not len(dof_indices):
            return np.zeros((self.len_ind(ind), 0))

        self._compute_bins()
        block_inds = np.digitize(dof_indices, self._bins) - 1
        dof_indices = dof_indices - self._bins[block_inds]
        block_inds = self._bin_map[block_inds]
        blocks = [_indexed(b, ind) for b in self._blocks]
        return np.array([blocks[bi].dofs([ci])[:, 0]
                         for bi, ci in zip(block_inds, dof_indices)]).T

    def amax(self, ind):
        self._compute_bins()
        blocks = [_indexed(b, ind) for b in self._blocks]
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


class BlockVectorArray(VectorArray):
    """|VectorArray| where each vector is a direct sum of sub-vectors.

    Given a list of equal length |VectorArrays| `blocks`, this |VectorArray|
    represents the direct sums of the vectors contained in the arrays.
    The associated |VectorSpace| is :class:`BlockVectorSpace`.

    :class:`BlockVectorArray` can be used in conjunction with
    :class:`~pymor.operators.block.BlockOperator`.
    """

    impl_type = BlockVectorArrayImpl

    def block(self, ind, copy=True):
        """Return a copy of a single block or a sequence of blocks."""
        if isinstance(ind, (tuple, list)):
            import warnings
            warnings.warn('Calling BlockVectorArray.block with a sequence is deprecated', DeprecationWarning)
            assert all(isinstance(ii, Number) for ii in ind)
            return tuple(self.block(ii) for ii in ind)

        assert isinstance(ind, Number)
        b = _indexed(self.impl._blocks[ind], self.ind)
        return b.copy() if copy else b

    @property
    def num_blocks(self):
        return len(self.space.subspaces)


class BlockVectorSpace(VectorSpace):
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

    def __init__(self, subspaces):
        subspaces = tuple(subspaces)
        assert all([isinstance(subspace, VectorSpace) for subspace in subspaces])
        self.subspaces = subspaces

    def __eq__(self, other):
        return (type(other) is BlockVectorSpace
                and len(self.subspaces) == len(other.subspaces)
                and all(space == other_space for space, other_space in zip(self.subspaces, other.subspaces)))

    def __hash__(self):
        return sum(hash(s) for s in self.subspaces)

    @property
    def dim(self):
        return sum(subspace.dim for subspace in self.subspaces)

    def zeros(self, count=1, reserve=0):
        # these asserts make sure we also trigger if the subspace list is empty
        assert count >= 0
        assert reserve >= 0
        return BlockVectorArray(
            self,
            BlockVectorArrayImpl([subspace.zeros(count=count, reserve=reserve) for subspace in self.subspaces], self)
        )

    @classinstancemethod
    def make_array(cls, obj):
        assert len(obj) > 0
        return cls(tuple(o.space for o in obj)).make_array(obj)

    @make_array.instancemethod
    def make_array(self, obj):
        """:noindex:"""
        assert len(obj) == len(self.subspaces)
        assert all(block in subspace for block, subspace in zip(obj, self.subspaces))
        return BlockVectorArray(self, BlockVectorArrayImpl(obj, self))

    def make_block_diagonal_array(self, obj):
        assert len(obj) == len(self.subspaces)
        assert all(block in subspace for block, subspace in zip(obj, self.subspaces))
        U = self.empty(reserve=sum(len(UU) for UU in obj))
        for i, UU in enumerate(obj):
            U.append(self.make_array([s.zeros(len(UU)) if j != i else UU for j, s in enumerate(self.subspaces)]))
        return U

    def from_numpy(self, data, ensure_copy=False):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data_ind = np.cumsum([0] + [subspace.dim for subspace in self.subspaces])
        return BlockVectorArray(
            self,
            BlockVectorArrayImpl([subspace.from_numpy(data[:, data_ind[i]:data_ind[i + 1]], ensure_copy=ensure_copy)
                                  for i, subspace in enumerate(self.subspaces)],
                                 self)
        )


def _indexed(block, ind):
    return block if ind is None else block[ind]
