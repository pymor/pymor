# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from pymor.algorithms.rules import RuleTable, match_class
from pymor.operators.block import BlockOperatorBase
from pymor.operators.constructions import (AdjointOperator, ComponentProjection, Concatenation, IdentityOperator,
                                           LincombOperator, VectorArrayOperator, ZeroOperator)
from pymor.operators.numpy import NumpyMatrixOperator


def to_matrix(op, format=None, mu=None):
    """Transform construction of Operators to matrix

    Parameters
    ----------
    op
        Operator.
    format
        Format of the resulting matrix: |NumPy array| if 'dense',
        otherwise the appropriate |SciPy spmatrix|.
        If `None`, a choice between dense and sparse format is
        automatically made.
    mu
        |Parameter|.

    Returns
    -------
    res
        Equivalent matrix.
    """
    assert format is None or format in ('dense', 'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil')
    op = op.assemble(mu)
    return ToMatrixRules(format, mu).apply(op)


class ToMatrixRules(RuleTable):

    def __init__(self, format, mu):
        super().__init__()
        self.format, self.mu = format, mu

    @match_class(NumpyMatrixOperator)
    def action_NumpyMatrixOperator(self, op):
        format = self.format
        if format is None:
            return op.matrix
        elif format == 'dense':
            if not op.sparse:
                return op.matrix
            else:
                return op.matrix.toarray()
        else:
            if not op.sparse:
                return getattr(sps, format + '_matrix')(op.matrix)
            else:
                return op.matrix.asformat(format)

    @match_class(BlockOperatorBase)
    def action_BlockOperator(self, op):
        format = self.format
        op_blocks = op._blocks
        mat_blocks = [[] for i in range(op.num_range_blocks)]
        is_dense = True
        for i in range(op.num_range_blocks):
            for j in range(op.num_source_blocks):
                mat_ij = self.apply(op_blocks[i, j])
                if sps.issparse(mat_ij):
                    is_dense = False
                mat_blocks[i].append(mat_ij)
        if format is None and is_dense or format == 'dense':
            return np.asarray(np.bmat(mat_blocks))
        else:
            return sps.bmat(mat_blocks, format=format)

    @match_class(AdjointOperator)
    def action_AdjointOperator(self, op):
        format = self.format
        res = self.apply(op.operator).T.conj()
        if op.range_product is not None:
            range_product = self.apply(op.range_product)
            if format is None and not sps.issparse(res) and sps.issparse(range_product):
                res = range_product.T.dot(res.T).T
            else:
                res = res.dot(range_product)
        if op.source_product is not None:
            source_product = self.apply(op.source_product)
            if not sps.issparse(source_product):
                res = spla.solve(source_product, res)
            else:
                res = spsla.spsolve(source_product, res)
        if format is not None and format != 'dense':
            res = getattr(sps, format + '_matrix')(res)
        return res

    @match_class(ComponentProjection)
    def action_ComponentProjection(self, op):
        format = self.format
        if format == 'dense':
            res = np.zeros((op.range.dim, op.source.dim))
            for i, j in enumerate(op.components):
                res[i, j] = 1
        else:
            data = np.ones((op.range.dim,))
            i = np.arange(op.range.dim)
            j = op.components
            res = sps.coo_matrix((data, (i, j)), shape=(op.range.dim, op.source.dim))
            res = res.asformat(format if format else 'csc')
        return res

    @match_class(Concatenation)
    def action_Concatenation(self, op):
        mats = [self.apply(o) for o in op.operators]
        while len(mats) > 1:
            if self.format is None and not sps.issparse(mats[-2]) and sps.issparse(mats[-1]):
                mats = mats[:-2] + [mats[-1].T.dot(mats[-2].T).T]
            else:
                mats = mats[:-2] + [mats[-2].dot(mats[-1])]
        return mats[0]

    @match_class(IdentityOperator)
    def action_IdentityOperator(self, op):
        format = self.format
        if format == 'dense':
            return np.eye(op.source.dim)
        else:
            return sps.eye(op.source.dim, format=format if format else 'csc')

    @match_class(LincombOperator)
    def action_LincombOperator(self, op):
        op_coefficients = op.evaluate_coefficients(self.mu)
        res = op_coefficients[0] * self.apply(op.operators[0])
        for i in range(1, len(op.operators)):
            res = res + op_coefficients[i] * self.apply(op.operators[i])
        return res

    @match_class(VectorArrayOperator)
    def action_VectorArrayOperator(self, op):
        format = self.format
        res = op._array.conj().to_numpy() if op.adjoint else op._array.to_numpy().T
        if format is not None and format != 'dense':
            res = getattr(sps, format + '_matrix')(res)
        return res

    @match_class(ZeroOperator)
    def action_ZeroOperator(self, op):
        format = self.format
        if format is None:
            return sps.csc_matrix((op.range.dim, op.source.dim))
        elif format == 'dense':
            return np.zeros((op.range.dim, op.source.dim))
        else:
            return getattr(sps, format + '_matrix')((op.range.dim, op.source.dim))
