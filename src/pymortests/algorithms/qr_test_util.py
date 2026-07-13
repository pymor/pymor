from warnings import warn

import numpy as np
import pytest
from scipy.linalg import hilbert

from pymor.algorithms.basic import almost_equal
from pymor.core.config import is_scipy_mkl
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray, VectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace

# use lower tolerance for MKL
if is_scipy_mkl():
    TOL = {'atol': 1e-12, 'rtol': 1e-13}
else:
    TOL = {'atol': 1e-13, 'rtol': 1e-13}


def generate_hilbert_va(vector_space_type: VectorSpace, n: int) -> VectorArray:
    """Generates a Hilbert matrix as a VectorArray of the specified VectorSpace type.

    Creates a in exact arithmetic full-rank, potentially ill-conditioned,
    square Hilbert matrix of the given dimension `n`.
    Larger hilbert matrices are more ill-conditioned,
    but are never rank-deficient in exact arithmetic
    """
    return vector_space_type(n).from_numpy(hilbert(n))


def evaluate_qr_empty(qr_method, qr_kwargs: dict={}):
    """Test a `qr_method` for an empty VectorArray.

    QR implementations have logic in place to catch this case.
    Ensure that the logic is correct.
    """
    n = 5
    V = NumpyVectorSpace(n).empty(0)
    Q, R = qr_method(V, return_R=True, copy=True, **qr_kwargs)
    assert isinstance(Q, VectorArray)
    assert len(Q) == 0
    assert V.space == Q.space
    assert isinstance(R, np.ndarray)
    assert R.shape == (0,0)

    Q2 = qr_method(V, return_R=False, copy=False, **qr_kwargs)
    assert isinstance(Q2, VectorArray)
    assert Q2 == V
    assert len(Q2) == 0


def evaluate_qr(qr_method, A: VectorArray, product: Operator, return_R: bool, copy: bool, qr_kwargs: dict={}):
    r"""Tests a `qr_method` for given parameters.

    This test is to ensure that the arguments `product`, `return_R`, `copy`
    and potential method specific arguments in `qr_kwargs` are working as intended.
    """
    # create copy of input matrix and work with it
    CPY = A.copy()

    tup = qr_method(A=CPY, product=product, return_R=return_R, copy=copy, **qr_kwargs)

    # check number outputs, their types and dimensions
    if return_R:
        assert isinstance(tup, tuple)
        Q, R = tup
        assert isinstance(R, np.ndarray)
        assert len(R.shape) == 2
        assert R.shape == (len(Q), len(A))
    else:
        Q = tup
    assert isinstance(Q, VectorArray)
    assert Q.space == A.space

    # check if copies work properly, i.e., if we have a copy A should not change and, vise-versa,
    # if we do not have a copy A should be overwritten by Q
    if copy:
        assert np.all(almost_equal(A, CPY, **TOL)), 'Reference copied from might be overwritten.'
    else:
        assert np.all(almost_equal(Q, CPY, **TOL)), 'In-place QR decomposition did not modifiy its input completly.'

    # check if solution has low errors
    # check loss of orthogonality
    assert np.allclose(Q.inner(Q, product=product), np.eye(len(Q)), **TOL), 'Too high Loss of Orthogonality'
    if return_R:
        # check reconstruction error
        assert np.all(almost_equal(A, Q.lincomb(R), **TOL)), 'Too high Reconstruction error'
        # check Cholesky error
        assert np.allclose(A.inner(A, product=product), (R.conj().T) @ R, **TOL),\
            'Too high Cholesky error ($A^H A - R^H R$)'

    # check if solution is deterministic
    Q2, R2 = qr_method(A=A, product=product, return_R=True, copy=False, **qr_kwargs)
    try:
        assert np.all(almost_equal(Q, Q2, **TOL))
        if return_R:
            assert np.allclose(R, R2, **TOL)
    except AssertionError:
        precision = np.get_printoptions()['precision']
        np.set_printoptions(precision=2)

        L = [f'Q\n{Q.to_numpy()}', f'Q2\n{Q2.to_numpy()}']
        if return_R:
            L += [f'R\n{R}', f'R2\n{R2}']
        warn('\n'.join(L))

        np.set_printoptions(precision=precision)

        pytest.xfail('QR method not deterministic.')
