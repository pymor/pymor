# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from pymor.algorithms.rules import RuleTable, match_class
from pymor.operators.block import BlockOperator
from pymor.operators.constructions import (AdjointOperator, ComponentProjection, Concatenation, IdentityOperator,
                                           LincombOperator, VectorArrayOperator, ZeroOperator)
from pymor.operators.numpy import NumpyMatrixOperator


def to_matrix(op, format=None, mu=None):
    """Transfrom construction of NumpyMatrixOperators to NumPy or SciPy array

    Parameters
    ----------
    op
        Operator.
    format
        Format of the resulting |SciPy spmatrix|.
        If `None`, a dense format is used.
    mu
        |Parameter|.

    Returns
    -------
    res
        Equivalent matrix.
    """
    assert format is None or format in ('bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil')
    op = op.assemble(mu)
    mapping = {
        'bsr': sps.bsr_matrix,
        'coo': sps.coo_matrix,
        'csc': sps.csc_matrix,
        'csr': sps.csr_matrix,
        'dia': sps.dia_matrix,
        'dok': sps.dok_matrix,
        'lil': sps.lil_matrix
    }
    return ToMatrixRules.apply(op, format, mapping, mu)


class ToMatrixRules(RuleTable):

    @match_class(NumpyMatrixOperator)
    def NumpyMatrixOperator(self, op, format, mapping, mu):
        if format is None:
            if not op.sparse:
                return op._matrix
            else:
                return op._matrix.toarray()
        else:
            return mapping[format](op._matrix)

    @match_class(BlockOperator)
    def BlockOperator(self, op, format, mapping, mu):
        op_blocks = op._blocks
        mat_blocks = [[] for i in range(op.num_range_blocks)]
        for i in range(op.num_range_blocks):
            for j in range(op.num_source_blocks):
                if op_blocks[i, j] is None:
                    if format is None:
                        mat_blocks[i].append(np.zeros((op.range.subspaces[i].dim, op.source.subspaces[j].dim)))
                    else:
                        mat_blocks[i].append(None)
                else:
                    mat_blocks[i].append(self.apply(op_blocks[i, j], format, mapping, mu))
        if format is None:
            return np.bmat(mat_blocks)
        else:
            return sps.bmat(mat_blocks, format=format)

    @match_class(AdjointOperator)
    def AdjointOperator(self, op, format, mapping, mu):
        res = self.apply(op.operator, format, mapping, mu).T
        if op.range_product is not None:
            res = res.dot(self.apply(op.range_product, format, mapping, mu))
        if op.source_product is not None:
            if format is None:
                res = spla.solve(self.apply(op.source_product, format, mapping, mu), res)
            else:
                res = spsla.spsolve(self.apply(op.source_product, format, mapping, mu), res)
        return res

    @match_class(ComponentProjection)
    def ComponentProjection(self, op, format, mapping, mu):
        if format is None:
            res = np.zeros((op.range.dim, op.source.dim))
            for i, j in enumerate(op.components):
                res[i, j] = 1
        else:
            data = np.ones((op.range.dim,))
            i = np.arange(op.range.dim)
            j = op.components
            res = sps.coo_matrix((data, (i, j)), shape=(op.range.dim, op.source.dim))
            res = res.asformat(format)
        return res

    @match_class(Concatenation)
    def Concatenation(self, op, format, mapping, mu):
        return self.apply(op.second, format, mapping, mu).dot(self.apply(op.first, format, mapping, mu))

    @match_class(IdentityOperator)
    def IdentityOperator(self, op, format, mapping, mu):
        if format is None:
            return np.eye(op.source.dim)
        else:
            return sps.eye(op.source.dim, format=format)

    @match_class(LincombOperator)
    def LincombOperator(self, op, format, mapping, mu):
        op_coefficients = op.evaluate_coefficients(mu)
        res = op_coefficients[0] * self.apply(op.operators[0], format, mapping, mu)
        for i in range(1, len(op.operators)):
            res = res + op_coefficients[i] * self.apply(op.operators[i], format, mapping, mu)
        return res

    @match_class(VectorArrayOperator)
    def VectorArrayOperator(self, op, format, mapping, mu):
        res = op._array.data if op.transposed else op._array.data.T
        if format is not None:
            res = mapping[format](res)
        return res

    @match_class(ZeroOperator)
    def ZeroOperator(self, op, format, mapping, mu):
        if format is None:
            return np.zeros((op.range.dim, op.source.dim))
        else:
            return mapping[format]((op.range.dim, op.source.dim))
