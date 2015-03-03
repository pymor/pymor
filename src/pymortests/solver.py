# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import diags
import pytest

from pymor.operators.basic import OperatorBase
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray


class GenericOperator(OperatorBase):

    source = range = NumpyVectorSpace(10)
    op = NumpyMatrixOperator(np.eye(10) * np.arange(1, 11))
    linear = True

    def apply(self, U, ind=None, mu=None):
        return self.op.apply(U, ind=ind, mu=mu)

    def apply_adjoint(self, U, ind=None, mu=None):
        return self.op.apply_adjoint(U, ind=ind, mu=mu)


@pytest.fixture(params=GenericOperator().invert_options.keys())
def generic_solver(request):
    return request.param


@pytest.fixture(params=NumpyMatrixOperator(np.eye(10)).invert_options.keys())
def numpy_dense_solver(request):
    return request.param


@pytest.fixture(params=NumpyMatrixOperator(diags([np.ones(10)], [0])).invert_options.keys())
def numpy_sparse_solver(request):
    return request.param


def test_generic_solvers(generic_solver):
    op = GenericOperator()
    rhs = NumpyVectorArray(np.ones(10))
    solution = op.apply_inverse(rhs, options=generic_solver)
    assert ((op.apply(solution) - rhs).l2_norm() / rhs.l2_norm())[0] < 1e-8


def test_numpy_dense_solvers(numpy_dense_solver):
    op = NumpyMatrixOperator(np.eye(10) * np.arange(1, 11))
    rhs = NumpyVectorArray(np.ones(10))
    solution = op.apply_inverse(rhs, options=numpy_dense_solver)
    assert ((op.apply(solution) - rhs).l2_norm() / rhs.l2_norm())[0] < 1e-8


def test_numpy_sparse_solvers(numpy_sparse_solver):
    op = NumpyMatrixOperator(diags([np.arange(1., 11.)], [0]))
    rhs = NumpyVectorArray(np.ones(10))
    solution = op.apply_inverse(rhs, options=numpy_sparse_solver)
    assert ((op.apply(solution) - rhs).l2_norm() / rhs.l2_norm())[0] < 1e-8
