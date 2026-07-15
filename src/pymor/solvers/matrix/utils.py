# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import numpy as np
import scipy.linalg as spla

from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator


@defaults('value')
def mat_eqn_sparse_min_size(value=1000):
    """Returns minimal size for which a sparse solver will be used by default."""
    return value


def _chol(A):
    """Cholesky decomposition.

    This implementation uses SVD to compute the Cholesky factor (can be used for singular matrices).

    Parameters
    ----------
    A
        Symmetric positive semidefinite matrix as a |NumPy array|.

    Returns
    -------
    L
        Cholesky factor of A (in the sense that L * L^T approximates A).
    """
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    from pymor.bindings.scipy import svd_lapack_driver
    U, s, _ = spla.svd(A, lapack_driver=svd_lapack_driver())
    L = U * np.sqrt(s)
    return L


def _solve_lyap_lrcf_check_args(A, E, B, trans):
    assert isinstance(A, Operator)
    assert A.linear
    assert not A.parametric
    assert A.source == A.range
    if E is not None:
        assert isinstance(E, Operator)
        assert E.linear
        assert not E.parametric
        assert E.source == E.range
        assert E.source == A.source
    assert B in A.source


def _solve_lyap_dense_check_args(A, E, B, trans):
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    if E is not None:
        assert isinstance(E, np.ndarray)
        assert E.ndim == 2
        assert E.shape[0] == E.shape[1]
        assert E.shape[0] == A.shape[0]
    assert isinstance(B, np.ndarray)
    assert B.ndim == 2 #TODO: this was a bug in the old pymor
    assert not trans and B.shape[0] == A.shape[0] or trans and B.shape[1] == A.shape[0]


def _solve_ricc_check_args(A, E, B, C, R, S, trans):
    assert isinstance(A, Operator)
    assert A.linear
    assert not A.parametric
    assert A.source == A.range
    if E is not None:
        assert isinstance(E, Operator)
        assert E.linear
        assert not E.parametric
        assert E.source == E.range == A.source
    assert B in A.source
    assert C in A.source
    if R is not None:
        assert isinstance(R, np.ndarray)
        assert R.ndim == 2
        assert R.shape[0] == R.shape[1]
        if not trans:
            assert R.shape[0] == len(C)
        else:
            assert R.shape[0] == len(B)
    if S is not None:
        assert S in A.source
        if not trans:
            assert len(C) == len(S)
        else:
            assert len(B) == len(S)


def _solve_ricc_dense_check_args(A, E, B, C, R, S, trans):
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    if E is not None:
        assert isinstance(E, np.ndarray)
        assert E.ndim == 2
        assert E.shape[0] == E.shape[1]
        assert E.shape[0] == A.shape[0]
    assert isinstance(B, np.ndarray)
    assert isinstance(C, np.ndarray)
    assert B.shape[0] == A.shape[0]
    assert C.shape[1] == A.shape[0]
    if R is not None:
        assert isinstance(R, np.ndarray)
        assert R.ndim == 2
        assert R.shape[0] == R.shape[1]
        if not trans:
            assert R.shape[0] == C.shape[0]
        else:
            assert R.shape[0] == B.shape[1]
    if S is not None:
        assert isinstance(S, np.ndarray)
        if not trans:
            assert S.shape[1] == A.shape[0]
            assert S.shape[0] == C.shape[0]
        else:
            assert S.shape[0] == A.shape[0]
            assert S.shape[1] == B.shape[1]

def _check_lyapunov_args(A, E, B, trans):
    assert isinstance(A, Operator)
    assert A.linear
    assert not A.parametric
    assert A.source == A.range
    if E is not None:
        assert isinstance(E, Operator)
        assert E.linear
        assert not E.parametric
        assert E.source == E.range
        assert E.source == A.source
    assert B in A.source


def _check_riccati_args(A, E, B, C, R, S, trans):
    assert isinstance(A, Operator)
    assert A.linear
    assert not A.parametric
    assert A.source == A.range
    if E is not None:
        assert isinstance(E, Operator)
        assert E.linear
        assert not E.parametric
        assert E.source == E.range == A.source
    assert B in A.source
    assert C in A.source
    if R is not None:
        assert isinstance(R, np.ndarray)
        assert R.ndim == 2
        assert R.shape[0] == R.shape[1]
        if not trans:
            assert R.shape[0] == len(C)
        else:
            assert R.shape[0] == len(B)
    if S is not None:
        assert S in A.source
        if not trans:
            assert len(C) == len(S)
        else:
            assert len(B) == len(S)


def _parse_options(options, default_options, default_solver, default_least_squares_solver, least_squares):
    if options is None:
        options = default_options[default_least_squares_solver] if least_squares else default_options[default_solver]
    elif isinstance(options, str):
        options = default_options[options]
    else:
        assert 'type' in options
        assert options['type'] in default_options
        assert options.keys() <= default_options[options['type']].keys()
        user_options = options
        options = default_options[user_options['type']]
        options.update(user_options)

    if least_squares != ('least_squares' in options['type']):
        logger = getLogger('foo')
        if least_squares:
            logger.warning('Non-least squares solver selected for least squares problem.')
        else:
            logger.warning('Least squares solver selected for non-least squares problem.')

    return options
