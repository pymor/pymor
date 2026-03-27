
import numpy as np
import pytest
from hypothesis import assume

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.chol_qr import shifted_chol_qr
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.exceptions import AccuracyError
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray


def evaluate_qr(qr_method, A: VectorArray, product: Operator, return_R: bool, copy: bool, qr_kwargs: dict):
    r"""Calls a given `qr_method` to compute the QR factorization of `A`.

    This test test is to ensure that the arguments `product`, `return_R`, `copy`
    and potential method specific arguments in `qr_kwargs` are working as intended.
    """
    # create copy of input matrix and work with it
    CPY = A.copy()

    try:
        tup = qr_method(A=CPY, product=product, return_R=return_R, copy=copy, **qr_kwargs)
    except AccuracyError as e:
        # if pyMOR's gram_schmidt fails something is clearly wrong
        if qr_method == gram_schmidt:
            raise e

        # gram_schmidt should be able to handle anything
        # use it to check if matrices are linearly-dependent
        Qgs = gram_schmidt(A, product=product, copy=True)
        if len(Qgs) < len(A):
            pytest.xfail('Test matrix contains linearly dependent vectors. \
                          QR method was unable to compute a factorization.')
        raise e


    # check number outputs, their types and dimensions
    assert (isinstance(tup, tuple) and len(tup) == 2) if return_R else (isinstance(tup, VectorArray))
    Q = tup[0] if return_R else tup
    if return_R:
        assert len(tup) == 2
        R = tup[1]
        assert R is not None
        assert len(R.shape) == 2
        assert R.shape[0] == len(Q)

    TOL = {'atol': 1e-13, 'rtol': 1e-13}

    # check if copies work properly, i.e., if we have a copy A should not change and vise-versa,
    # if we do not have a copy, A should be overwritten by Q
    if copy:
        assert np.all(almost_equal(A, CPY, **TOL)), 'Reference copied from might be overwritten.'
    else:
        assert np.all(almost_equal(Q, CPY, **TOL)), 'In-place QR decomposition did not modifiy its input completly.'

    # check if solution has low errors
    assert np.allclose(Q.inner(Q, product=product), np.eye(len(Q)), **TOL), 'Q^H Q = I not fulfilled'
    if return_R:
        assert np.all(almost_equal(A, Q.lincomb(R), **TOL)), 'QR = A not fulfilled'
        assert np.allclose(
            A.inner(A, product=product), (R.conj().T) @ R, **TOL
        ), 'A^H A = R^H Q^H Q R = R^H R not fulfilled'

    # check if solution is deterministic
    Q2, R2 = qr_method(A=A, product=product, return_R=True, copy=False, **qr_kwargs)
    assert np.all(almost_equal(Q, Q2, **TOL))
    if return_R:
        assert np.allclose(R, R2, **TOL)


def evaluate_qr_offset(qr_method, A: VectorArray, num_blocks: int, qr_kwargs: dict, atol=1e-13, rtol=1e-13):
    r"""Performs QR-Update using `qr_method` onto `num_blocks` many blocks of matrix `A`.

    This test test is to ensure that the offset is working as intended.
    """
    assume(len(A) > 0 and A.dim > 0)

    try:
        # create copy of input matrix and work with it
        n = len(A)
        if num_blocks <= 0:
            num_blocks = n

        if num_blocks > n:
            return # skip

        CPY = A.copy()
        Q = CPY.space.empty()
        off_R = off_Q = 0

        block_size = int(np.ceil(n/num_blocks))
        R = np.zeros([n,n])

        i = 0
        while len(CPY) > 0:
            i += 1
            min_num = min(len(CPY), block_size)
            # the way CholQR with offset is implemented in pyMOR,
            # it deletes and appends vectors in each iteration
            # therefore, we cannot start with all vectors but have to append them for each call
            off_Q = len(Q)
            Q.append(CPY[:min_num])
            del CPY[:min_num]

            _, R_ = qr_method(Q, offset=off_Q, return_R=True, copy=False, **qr_kwargs)
            R = R.astype(np.promote_types(R.dtype, R_.dtype), copy=False)

            h, w = R_[:,off_Q:].shape
            R[:h, off_R:off_R+w] = R_[:,off_Q:]
            off_R += w

        # vectors might be removed and R therefore not upper-triangular, but upper-trapezoidal
        R = R[:len(Q),:]

        # check if solution has low errors
        assert np.allclose(Q.inner(Q), np.eye(len(Q)), rtol, atol), 'Q^H Q = I not fulfilled'
        assert np.all(almost_equal(A, Q.lincomb(R), atol=atol, rtol=rtol)), 'QR = A not fulfilled'
        assert np.allclose(A.inner(A), R.conj().T @ R, rtol, atol), 'A^H A = R^H Q^H Q R = R^H R not fulfilled'
    except (AccuracyError, AssertionError) as e:
        # if pyMOR's gram_schmidt fails something is clearly wrong
        if qr_method == gram_schmidt:
            raise e

        if qr_method == shifted_chol_qr and 'recompute_shift' in qr_kwargs:
            # flag recompute_shift might be in qr_kwargs, but set to False
            if qr_kwargs['recompute_shift']:
                raise e

        # gram_schmidt should be able to handle anything
        # use it to check if matrices are linearly-dependent
        Qgs = gram_schmidt(A, copy=True)
        if len(Qgs) < len(A):
            pytest.xfail('Test matrix contains linearly dependent vectors. \
                          QR method was unable to compute a factorization. {e}')
        raise e
