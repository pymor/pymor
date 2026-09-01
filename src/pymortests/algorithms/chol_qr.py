# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import pytest
from hypothesis import assume
from hypothesis.errors import UnsatisfiedAssumption

from pymor.algorithms.chol_qr import shifted_chol_qr
from pymor.core.config import is_scipy_mkl
from pymor.vectorarrays.list import NumpyListVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.algorithms.qr_test_util import (
    evaluate_qr,
    evaluate_qr_empty,
    evaluate_qr_full_offset,
    evaluate_qr_offset,
    generate_hilbert_va,
)
from pymortests.base import runmodule

# use lower tolerance for MKL
ORTH_TOL = 1e-13 if is_scipy_mkl() else 5e-15


def test_chol_qr_empty():
    evaluate_qr_empty(shifted_chol_qr)


def test_chol_qr_full_offset():
    evaluate_qr_full_offset(shifted_chol_qr)


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


# into how many blocks the matrix should be split; 0 for n blocks/ single vectors
@pytest.mark.parametrize('num_blocks', [1, 2, 7, 0])
@pytest.mark.parametrize('recompute_shift', [False, True])
def test_chol_qr_with_offset(num_blocks, recompute_shift):
    A = generate_hilbert_va(NumpyVectorSpace, 251)
    evaluate_qr_offset(shifted_chol_qr, A, num_blocks,
        {'recompute_shift': recompute_shift, 'maxiter': 10, 'orth_tol': ORTH_TOL}
    )


if __name__ == '__main__':
    runmodule(filename=__file__)
