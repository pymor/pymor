# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from numpy.random import uniform

from pymor.algorithms.randrangefinder import rrf, adaptive_rrf
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import VectorArrayOperator


np.random.seed(0)
A = uniform(low=-1.0, high=1.0, size=(100, 100))
A = A.dot(A.T)
range_product = NumpyMatrixOperator(A)

np.random.seed(1)
A = uniform(low=-1.0, high=1.0, size=(10, 10))
A = A.dot(A.T)
source_product = NumpyMatrixOperator(A)

B = range_product.range.random(10, seed=10)
op = VectorArrayOperator(B)

C = range_product.range.random(10, seed=11)+1j*range_product.range.random(10, seed=12)
op_complex = VectorArrayOperator(C)


def test_rrf():
    Q = rrf(op, source_product, range_product)
    assert Q in op.range
    assert len(Q) == 8

    Q = rrf(op_complex, iscomplex=True)
    assert np.iscomplexobj(Q.to_numpy())
    assert Q in op.range
    assert len(Q) == 8


def test_adaptive_rrf():
    B = adaptive_rrf(op, source_product, range_product)
    assert B in op.range

    B = adaptive_rrf(op_complex, iscomplex=True)
    assert np.iscomplexobj(B.to_numpy())
    assert B in op.range
