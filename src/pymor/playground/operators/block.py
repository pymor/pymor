# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import coo_matrix, bmat

from pymor.operators.interfaces import OperatorInterface
from pymor.la.interfaces import VectorSpace
from pymor.la.numpyvectorarray import NumpyVectorArray
from pymor.playground.la.blockvectorarray import BlockVectorArray
from pymor.operators.basic import NumpyMatrixOperator, OperatorBase


class BlockOperator(OperatorBase):
    """A sparse matrix of arbitrary operators
    """

    def _operators(self):
        for row in self._blocks:
            for entry in row:
                if entry is not None:
                    yield entry

    def _enumerated_operators(self):
        for ii, row in enumerate(self._blocks):
            for jj, entry in enumerate(row):
                yield ii, jj, entry

    def __init__(self, blocks, sources=None, ranges=None):
        # check input
        assert isinstance(blocks, list)
        if isinstance(blocks[0], list):
            self._blocks = blocks
        else:
            self._blocks = []
            self._blocks.append(blocks)
        # each block has to be an operator or None
        assert all([isinstance(op, OperatorInterface) for op in self._operators()])
        # all rows of blocks have to have the same lenght
        assert all([len(row) == len(self._blocks[0]) for row in self._blocks])
        # all rows need to have at least one operator
        assert all([any([op is not None for op in row]) for row in self._blocks])
        # all columns need to have at least one operator
        assert all([any([row[ii] is not None for row in self._blocks]) for ii in np.arange(len(self._blocks[0]))])
        # build source and range
        source_types = [None for ii in np.arange(len(self._blocks[0]))]
        range_types  = [None for jj in np.arange(len(self._blocks))]
        for ii, jj, op in self._enumerated_operators():
            if op is not None:
                assert source_types[jj] is None or op.source == source_types[jj]
                source_types[jj] = op.source
                assert range_types[ii] is None or op.range == range_types[ii]
                range_types[ii] = op.range
        # there needs to be at least one operator for each combination of row and column
        assert all([ss is not None for ss in source_types])
        assert all([rr is not None for rr in range_types])
        self.source = VectorSpace(BlockVectorArray, tuple(source_types))
        self.range = VectorSpace(BlockVectorArray, tuple(range_types))
        # some information
        self._source_dims = tuple(space.dim for space in self.source.subtype)
        self._range_dims  = tuple(space.dim for space in self.range.subtype)
        self.num_source_blocks = len(source_types)
        self.num_range_blocks  = len(range_types)
        self.linear = all([op.linear for op in self._operators()])
        self._is_diagonal = (all([block is None if ii != jj else True for ii, jj, block in self._enumerated_operators()])
                             and self.num_source_blocks == self.num_range_blocks)
        # build parameter type
        self.build_parameter_type(inherits=list(self._operators()))

    def apply(self, U, ind=None, mu=None):
        ind = [0] if ind is None else ind
        if len(U) == 0:
            return self.range.empty()
        if len(U) != 1:
            raise NotImplementedError
        if len(ind) != 1 or ind[0] != 0:
            raise NotImplementedError
        V = [range_type.zeros(len(ind)) for range_type in self.range.subtype]
        for ii, jj, op in self._enumerated_operators():
            if op is not None:
                V[ii] += op.apply(U._blocks[jj], ind=ind, mu=mu)
        return BlockVectorArray(V)

    def apply2(self, V, U, pairwise, U_ind=None, V_ind=None, mu=None, product=None):
        assert U in self.source
        assert V in self.range
        if self._is_diagonal and product is None:
            if len(U) != 1:
                raise NotImplementedError
            if len(V) != 1:
                raise NotImplementedError
            if U_ind is not None:
                raise NotImplementedError
            if V_ind is not None:
                raise NotImplementedError
            return sum([self._blocks[ii][ii].apply2(V._blocks[ii], U._blocks[ii], pairwise)
                        for ii in np.arange(self.num_source_blocks)])
        else:
            return super(BlockOperator, self).apply2(V, U, pairwise, U_ind=U_ind, V_ind=V_ind, mu=mu, product=product)

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        assembled = self.assemble(mu=mu)
        if isinstance(assembled, NumpyMatrixOperator):
            solution = assembled.apply_inverse(U, ind=ind, options=options)
            assert len(solution) == 1
            assert (not np.isnan(np.sum(solution._array))) and (not np.isinf(np.sum(solution._array)))
            block_sizes = [sp.subtype for sp in self.source.subtype]
            blocks = [NumpyVectorArray(solution._array[0][sum(block_sizes[:ss]):sum(block_sizes[:(ss + 1)])])
                      for ss in np.arange(len(block_sizes))]
            return BlockVectorArray(blocks)
        else:
            # TODO: implement use of generic solver
            raise NotImplementedError

    def projected(self, source_basis, range_basis, product=None, name=None):
        if product is not None:
            raise NotImplementedError
        assert source_basis is not None or range_basis is not None
        assert isinstance(source_basis, (tuple, list)) or source_basis is None
        assert isinstance(range_basis, (tuple, list)) or range_basis is None
        source_basis = tuple([None for ss in np.arange(self.num_source_blocks)]
                             if source_basis is None else source_basis)
        range_basis  = tuple([None for rr in np.arange(self.num_range_blocks)]
                             if range_basis  is None else range_basis)
        return BlockOperator([[self._blocks[ii][jj].projected(source_basis[jj], range_basis[ii], name=name)
                               if self._blocks[ii][jj] is not None else None
                               for jj in np.arange(self.num_source_blocks)]
                              for ii in np.arange(self.num_range_blocks)])

    def assemble(self, mu=None):
        # handle empty case
        dim_source = np.sum(self._source_dims)
        dim_range  = np.sum(self._range_dims)
        if dim_source == 0 or dim_range == 0:
            return NumpyMatrixOperator(np.zeros((dim_range, dim_source)))
        # TODO: convert mu to correct local mu for each block
        assembled_blocks = [[self._blocks[ii][jj].assemble(mu) if self._blocks[ii][jj] is not None else None
                             for jj in np.arange(self.num_source_blocks)]
                            for ii in np.arange(self.num_range_blocks)]
        if all(all([isinstance(op, NumpyMatrixOperator) if op is not None else True for op in row]) for row in assembled_blocks):
            mat = bmat([[coo_matrix(assembled_blocks[ii][jj]._matrix)
                         if assembled_blocks[ii][jj] is not None else coo_matrix((self._range_dims[ii], self._source_dims[jj]))
                         for jj in np.arange(self.num_source_blocks)]
                        for ii in np.arange(self.num_range_blocks)])
            # TODO: decide (depending on the size of mat) if we want to call todense()?
            return NumpyMatrixOperator(mat.todense())
        else:
            return BlockOperator(assembled_blocks)

    def as_vector(self, mu=None):
        if not self.linear:
            raise TypeError('This nonlinear operator does not represent a vector or linear functional.')
        elif self.source.dim == 1 and self.num_source_blocks == 1 and self.source.subtype[0].type is NumpyVectorArray:
            # we are a vector
            if not all(rr.type is NumpyVectorArray for rr in self.range.subtype):
                raise NotImplementedError
            return NumpyVectorArray(np.concatenate([self._blocks[ii][0].assemble(mu)._matrix
                                                    for ii in np.arange(self.num_range_blocks)],
                                                   axis=1))
        elif self.range.dim == 1 and self.num_range_blocks == 1 and self.range.subtype[0].type is NumpyVectorArray:
            # we are a functional
            if not all(ss.type is NumpyVectorArray for ss in self.source.subtype):
                raise NotImplementedError
            return NumpyVectorArray(np.concatenate([self._blocks[0][jj].assemble(mu)._matrix
                                                    for jj in np.arange(self.num_source_blocks)],
                                                   axis=1))
        else:
            raise TypeError('This operator does not represent a vector or linear functional.')

