# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

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
    return _to_matrix(op, format, mapping, mu)


def _to_matrix(op, format, mapping, mu):
    if isinstance(op, NumpyMatrixOperator):
        if format is None:
            if not op.sparse:
                res = op._matrix
            else:
                res = op._matrix.toarray()
        else:
            res = mapping[format](op._matrix)
    elif isinstance(op, BlockOperator):
        op_blocks = op._blocks
        mat_blocks = [[] for i in range(op.num_range_blocks)]
        for i in range(op.num_range_blocks):
            for j in range(op.num_source_blocks):
                if op_blocks[i, j] is None:
                    if format is None:
                        mat_blocks[i].append(np.zeros((op.range.subtype[i].dim, op.source.subtype[j].dim)))
                    else:
                        mat_blocks[i].append(None)
                else:
                    mat_blocks[i].append(_to_matrix(op_blocks[i, j], format, mapping, mu))
        if format is None:
            res = np.bmat(mat_blocks)
        else:
            res = sps.bmat(mat_blocks, format=format)
    elif isinstance(op, AdjointOperator):
        res = _to_matrix(op.operator, format, mapping, mu).T
        if op.range_product is not None:
            res = res.dot(_to_matrix(op.range_product, format, mapping, mu))
        if op.source_product is not None:
            if format is None:
                res = spla.solve(_to_matrix(op.source_product, format, mapping, mu), res)
            else:
                res = spsla.spsolve(_to_matrix(op.source_product, format, mapping, mu), res)
    elif isinstance(op, ComponentProjection):
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
    elif isinstance(op, Concatenation):
        res = _to_matrix(op.second, format, mapping, mu).dot(_to_matrix(op.first, format, mapping, mu))
    elif isinstance(op, IdentityOperator):
        if format is None:
            res = np.eye(op.source.dim)
        else:
            res = sps.eye(op.source.dim, format=format)
    elif isinstance(op, LincombOperator):
        op_coefficients = op.evaluate_coefficients(mu)
        res = op_coefficients[0] * _to_matrix(op.operators[0], format, mapping, mu)
        for i in range(1, len(op.operators)):
            res = res + op_coefficients[i] * _to_matrix(op.operators[i], format, mapping, mu)
    elif isinstance(op, VectorArrayOperator):
        res = op._array.data if op.transposed else op._array.data.T
        if format is not None:
            res = mapping[format](res)
    elif isinstance(op, ZeroOperator):
        if format is None:
            res = np.zeros((op.range.dim, op.source.dim))
        else:
            res = mapping[format]((op.range.dim, op.source.dim))
    else:
        raise ValueError('Encountered unsupported operator type {}'.format(type(op)))
    return res
