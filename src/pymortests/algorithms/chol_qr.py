# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import warnings

import numpy as np
import pytest
from hypothesis import assume
from hypothesis.errors import UnsatisfiedAssumption

import pymortests.strategies as pyst
from pymor.algorithms.basic import almost_equal, contains_zero_vector
from pymor.algorithms.chol_qr import shifted_chol_qr
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.config import is_scipy_mkl
from pymor.vectorarrays.list import NumpyListVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.algorithms.qr_test_util import evaluate_qr, evaluate_qr_empty, generate_hilbert_va
from pymortests.base import runmodule

# use lower tolerance for MKL
ORTH_TOL = 1e-13 if is_scipy_mkl() else 5e-15


def test_chol_qr_empty():
    evaluate_qr_empty(shifted_chol_qr)


@pytest.mark.parametrize('va_space', [NumpyVectorSpace, NumpyListVectorSpace])
@pytest.mark.parametrize('n', [1, 5, 25])
@pytest.mark.parametrize('return_R', [False, True])
@pytest.mark.parametrize('copy', [False, True])
@pytest.mark.parametrize('recompute_shift', [False, True])
@pytest.mark.parametrize('orth_tol', [None, ORTH_TOL])
@pytest.mark.parametrize('rtol', [0, 1e-13, 1e-8])
def test_chol_qr_parameters(va_space, n, return_R, copy, recompute_shift, orth_tol, rtol):
    A = generate_hilbert_va(va_space, n)
    evaluate_qr(shifted_chol_qr, A, None, return_R, copy,
        {'recompute_shift': recompute_shift, 'maxiter': 10, 'orth_tol': orth_tol, 'rtol': rtol}
    )


@pytest.mark.parametrize('recompute_shift', [False, True])
def test_chol_qr_with_product(operator_with_arrays_and_products, recompute_shift):
    _, _, A, _, product, _ = operator_with_arrays_and_products

    # keeps failing due to restriction; requires more hypothesis examples
    try:
        assume(A.dim >= len(A))
    except UnsatisfiedAssumption:
        pytest.xfail(f'{UnsatisfiedAssumption.__qualname__} was raised')

    evaluate_qr(shifted_chol_qr, A, product, True, True,
        {'recompute_shift': recompute_shift, 'maxiter': 10, 'orth_tol': ORTH_TOL}
    )


@pytest.mark.xfail
@pyst.given_vector_arrays()
def test_shifted_chol_qr_with_offset(vector_array):
    U = vector_array

    assume(len(U) >= 2 and not contains_zero_vector(U))

    # check if the test VectorArray contains linearly dependent vectors
    # Q and R are later used for comparison
    onbgs, Rgs = gram_schmidt(U, copy=True, return_R=True)
    if len(onbgs) < len(U):
        warnings.warn('Linearly dependent vectors detected! Skipping ...')
        return

    # orthogonalize first half
    offset = round(len(U) / 2)
    onb, Ro = shifted_chol_qr(U[:offset], return_R=True, copy=True)
    assert np.allclose(onb.inner(onb), np.eye(len(onb)))

    # orthogonalize second half
    onb.append(U[offset:])
    _, R = shifted_chol_qr(onb, return_R=True, copy=False)

    # insert R of first orthogonalization step into R of second step
    # overwritten matrix block only contains a unit matrix
    R[:offset,:offset] = Ro

    # compare Q and R of shifted Cholesky QR with offset to the gram_schmidt implementation
    rtol = atol = 1e-13
    assert np.all(almost_equal(onb, onbgs, rtol=rtol, atol=atol))
    assert np.allclose(Rgs, R)


if __name__ == '__main__':
    runmodule(filename=__file__)
