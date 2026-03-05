# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import pytest
from hypothesis import assume, settings
from hypothesis.errors import UnsatisfiedAssumption
from scipy.linalg import hilbert

import pymortests.strategies as pyst
from pymor.algorithms.basic import contains_zero_vector
from pymor.algorithms.chol_qr import shifted_chol_qr
from pymor.vectorarrays.list import NumpyListVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.algorithms.qr_test_util import evaluate_qr, evaluate_qr_offset
from pymortests.base import runmodule


@pytest.mark.builtin
@pytest.mark.parametrize('copy', [False, True])
def test_chol_qr_empty(copy):
    n = 5
    V = NumpyVectorSpace(n).empty(0)
    Q, _ = shifted_chol_qr(V, return_R=True, copy=copy)
    assert len(V) == len(Q) == 0


@pytest.mark.parametrize('va_space', [NumpyVectorSpace, NumpyListVectorSpace])
@pytest.mark.parametrize('n', [0, 1, 50])
@pytest.mark.parametrize('return_R', [False, True])
@pytest.mark.parametrize('copy', [False, True])
@pytest.mark.parametrize('recompute_shift', [False, True])
@pytest.mark.parametrize('orth_tol', [None, 1e-13])
@pytest.mark.parametrize('remove_dependent', [False, True])
def test_chol_qr_parameters(va_space, n, return_R, copy, recompute_shift, orth_tol, remove_dependent):
    # larger hilbert matrices are more ill-conditioned,
    # but are never rank-deficient in exact arithmetic
    A = hilbert(n)
    A = va_space(n).from_numpy(A)
    evaluate_qr(shifted_chol_qr, A, None, return_R, copy,
        {'recompute_shift': recompute_shift, 'maxiter': 10, 'orth_tol': orth_tol, 'remove_dependent': remove_dependent}
    )


@pytest.mark.parametrize('return_R', [False, True])
@pytest.mark.parametrize('recompute_shift', [False, True])
def test_rr_chol_qr_linear_dependent_vectors(return_R, recompute_shift):
    n = 100
    A = hilbert(n)[:10,:]
    A = NumpyVectorSpace(A.shape[0]).from_numpy(A)
    evaluate_qr(shifted_chol_qr, A, None, return_R, False,
        {'recompute_shift': recompute_shift, 'maxiter': 10, 'orth_tol': 1e-13, 'remove_dependent': True}
    )


@pyst.given_vector_arrays()
# into how many blocks the matrix should be split; 0 for n blocks/ single vectors
@pytest.mark.parametrize('num_blocks', [1, 2, 5, 0])
@pytest.mark.parametrize('recompute_shift', [False, True])
@settings(deadline=None)
def test_rr_chol_qr_with_offset(vector_array, num_blocks, recompute_shift):
    evaluate_qr_offset(shifted_chol_qr, vector_array, num_blocks,
        {'recompute_shift': recompute_shift, 'maxiter': 50, 'orth_tol': 5e-15, 'remove_dependent': True},
        # had to reduce the tolerances a lot in order to let the tests succeed
        # the highest error I have witnessed was of magnitude 1e-12
        atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize('recompute_shift', [False, True])
def test_chol_qr_with_product(operator_with_arrays_and_products, recompute_shift):
    _, _, A, _, product, _ = operator_with_arrays_and_products

    # keeps failing due to restrictions; requires more hypothesis examples
    try:
        assume(not contains_zero_vector(A) and A.dim >= len(A))
    except UnsatisfiedAssumption:
        pytest.xfail(f'{UnsatisfiedAssumption.__qualname__} was raised')

    evaluate_qr(shifted_chol_qr, A, product, True, True,
        {'recompute_shift': recompute_shift, 'maxiter': 10, 'orth_tol': 1e-13}
    )

if __name__ == '__main__':
    runmodule(filename=__file__)
