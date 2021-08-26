# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from functools import reduce
from numbers import Number
import numpy as np

from pymor.core.base import classinstancemethod
from pymor.vectorarrays.interface import VectorArray, VectorSpace


class BlockVectorArray(VectorArray):
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

    def to_numpy(self, ensure_copy=False):
        if len(self._blocks):
            # hstack will error out with empty input list
            return np.hstack([block.to_numpy() for block in self._blocks])
        else:
            return np.empty((0, 0))

    @property
    def real(self):
        return BlockVectorArray([block.real for block in self._blocks], self.space)

    @property
    def imag(self):
        return BlockVectorArray([block.imag for block in self._blocks], self.space)

    def conj(self):
        return BlockVectorArray([block.conj() for block in self._blocks], self.space)

    def block(self, ind):
        """Return a copy of a single block or a sequence of blocks."""
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
        try:
            return len(self._blocks[0])
        except IndexError:
            return 0

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
        assert isinstance(alpha, Number) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (len(self),)
        if len(x) > 0:
            for block, x_block in zip(self._blocks, x._blocks):
                block.axpy(alpha, x_block)
        else:
            assert len(self) == 0

    def inner(self, other, product=None):
        assert other in self.space
        if product is not None:
            return product.apply2(self, other)

        prods = [block.inner(other_block) for block, other_block in zip(self._blocks, other._blocks)]
        assert all([prod.shape == prods[0].shape for prod in prods])
        common_dtype = reduce(np.promote_types, (prod.dtype for prod in prods))
        ret = np.zeros(prods[0].shape, dtype=common_dtype)
        for prod in prods:
            ret += prod
        return ret

    def pairwise_inner(self, other, product=None):
        assert other in self.space
        if product is not None:
            return product.pairwise_apply2(self, other)

        prods = [block.pairwise_inner(other_block)
                 for block, other_block in zip(self._blocks, other._blocks)]
        assert all([prod.shape == prods[0].shape for prod in prods])
        common_dtype = reduce(np.promote_types, (prod.dtype for prod in prods))
        ret = np.zeros(prods[0].shape, dtype=common_dtype)
        for prod in prods:
            ret += prod
        return ret

    def lincomb(self, coefficients):
        lincombs = [block.lincomb(coefficients) for block in self._blocks]
        return BlockVectorArray(lincombs, self.space)

    def _norm(self):
        return np.sqrt(self.norm2())

    def _norm2(self):
        return np.sum(np.array([block.norm2() for block in self._blocks]), axis=0)

    def sup_norm(self):
        return np.max(np.array([block.sup_norm() for block in self._blocks]), axis=0)

    def dofs(self, dof_indices):
        dof_indices = np.array(dof_indices)
        if not len(dof_indices):
            return np.zeros((len(self), 0))

        self._compute_bins()
        block_inds = np.digitize(dof_indices, self._bins) - 1
        dof_indices -= self._bins[block_inds]
        block_inds = self._bin_map[block_inds]
        blocks = self._blocks
        return np.array([blocks[bi].dofs([ci])[:, 0]
                         for bi, ci in zip(block_inds, dof_indices)]).T

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
        return BlockVectorArray([subspace.zeros(count=count, reserve=reserve) for subspace in self.subspaces], self)

    @classinstancemethod
    def make_array(cls, obj):
        assert len(obj) > 0
        return cls(tuple(o.space for o in obj)).make_array(obj)

    @make_array.instancemethod
    def make_array(self, obj):
        """:noindex:"""
        assert len(obj) == len(self.subspaces)
        assert all(block in subspace for block, subspace in zip(obj, self.subspaces))
        return BlockVectorArray(obj, self)

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
        return BlockVectorArray([subspace.from_numpy(data[:, data_ind[i]:data_ind[i + 1]], ensure_copy=ensure_copy)
                                 for i, subspace in enumerate(self.subspaces)], self)


class BlockVectorArrayView(BlockVectorArray):

    is_view = True

    def __init__(self, base, ind):
        self._blocks = tuple(block[ind] for block in base._blocks)
        self.space = base.space
