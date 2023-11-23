# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.operators.interface import Operator
from pymor.tools.frozendict import FrozenDict

_DEFAULT_LYAP_SOLVER_BACKEND = FrozenDict(
    {
        'cont': FrozenDict(
            {
                'sparse': 'lradi',
                'dense': 'slycot'
                if config.HAVE_SLYCOT
                else 'pymepack'
                if config.HAVE_PYMEPACK
                else 'scipy',
            }
        ),
        'disc': FrozenDict({'dense': 'slycot' if config.HAVE_SLYCOT else 'scipy'}),
    }
)


@defaults('value')
def mat_eqn_sparse_min_size(value=1000):
    """Returns minimal size for which a sparse solver will be used by default."""
    return value


@defaults('default_sparse_solver_backend', 'default_dense_solver_backend')
def solve_cont_lyap_lrcf(A, E, B, Q=None, S=None, hess=False, trans=False, options=None,
                         default_sparse_solver_backend=_DEFAULT_LYAP_SOLVER_BACKEND['cont']['sparse'],
                         default_dense_solver_backend=_DEFAULT_LYAP_SOLVER_BACKEND['cont']['dense']):
    """Compute an approximate low-rank solution of a continuous-time Lyapunov equation.

    Returns a low-rank Cholesky factor :math:`Z` such that :math:`Z Z^T` approximates the solution
    :math:`X` of a (generalized) continuous-time algebraic Lyapunov equation:

    - if trans is `False` and E is `None`:

      .. math::
         A X + X A^T + B B^T = 0,

    - if trans is `False` and E is an |Operator|:

      .. math::
          A X E^T + E X A^T + B B^T = 0,

    - if trans is `True` and E is `None`:

      .. math::
          A^T X + X A + B^T B = 0,

    - if trans is `True` and E is an |Operator|:

      .. math::
          A^T X E + E^T X A + B^T B = 0.

    - if hess is `True` and E is `None`, A is in upper Hessenberg form.

    - if hess is `True` and E is an |Operator|, (A, E) is in generalized Hessenberg form.


    We assume A and E are real |Operators|, E is invertible, and all the eigenvalues of (A, E) all
    lie in the open left half-plane. Operator B needs to be given as a |VectorArray| from
    `A.source`, and for large-scale problems, we assume `len(B)` is small.

    Q and S are real |Operators|. If they are given, they contain left and right Schur vectors of (A, E)
    respectively. In this case, A contains its upper triangular matrix U and E contains its upper
    triangular matrix V of the generalized Schur decomposition of (A, E),
    where :math:`A = Q U S^T` and :math:`E = Q V S^T` .

    If Q is given and S is `None`, E is supposed to be `None` as well. In this case, A contains its
    upper triangular matrix U beeing the Schur decomposition of A, Q contains the Schur vectors of A.
    :math:`A = Q U Q^T` .

    Parameters `Q`, `S` and `hess` are supported only by pymepack solvers. When another solver is
    selected, `NotImplementedError` will be raised.

    If the solver is not specified using the options argument, a solver backend is chosen based on
    availability in the following order:

    - for sparse problems (minimum size specified by
      :func:`mat_eqn_sparse_min_size`)

      1. `lradi` (see :func:`pymor.algorithms.lradi.solve_lyap_lrcf`),

    - for dense problems (smaller than :func:`mat_eqn_sparse_min_size`)

      1. `slycot` (see :func:`pymor.bindings.slycot.solve_lyap_lrcf`),
      2. `scipy` (see :func:`pymor.bindings.scipy.solve_lyap_lrcf`),
      3. `pymepack` (see :func:`pymor.bindings.pymepack.solve_lyap_lrcf`).

    Parameters
    ----------
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    Q
        The non-parametric |Operator| Q or `None`.
    S
        The non-parametric |Operator| S or `None`.
    hess
        Whether A or (A, E) is in the (generalized) Hessenberg form.
    trans
        Whether the first |Operator| in the Lyapunov equation is transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.algorithms.lradi.lyap_lrcf_solver_options`,
        - :func:`pymor.bindings.scipy.lyap_lrcf_solver_options`,
        - :func:`pymor.bindings.slycot.lyap_lrcf_solver_options`,
        - :func:`pymor.bindings.pymess.lyap_lrcf_solver_options`,
        - :func:`pymor.bindings.pymepack.lyap_lrcf_solver_options`.

    default_sparse_solver_backend
        Default sparse solver backend to use (lradi, pymess).
    default_dense_solver_backend
        Default dense solver backend to use (slycot, pymepack, scipy, pymess).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Lyapunov equation solution, |VectorArray| from `A.source`.
    """
    _solve_lyap_lrcf_check_args(A, E, B, trans, Q, S)
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        if A.source.dim >= mat_eqn_sparse_min_size():
            backend = default_sparse_solver_backend
        else:
            backend = default_dense_solver_backend
    if backend != 'pymepack':
        if (Q is not None or S is not None) :
            raise NotImplementedError(f'Backend solver ({backend}) does not support Schur factorization.')
        if hess:
            raise NotImplementedError(f'Backend solver ({backend}) does not support Hessenberg form.')
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'pymess':
        from pymor.bindings.pymess import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'lradi':
        from pymor.algorithms.lradi import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'pymepack':
        from pymor.bindings.pymepack import solve_lyap_lrcf as solve_lyap_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    if backend == 'pymepack':
        return solve_lyap_impl(A, E, B, Q=Q, S=S, hess=hess, trans=trans, cont_time=True, options=options)
    return solve_lyap_impl(A, E, B, trans=trans, cont_time=True, options=options)


@defaults('default_dense_solver_backend')
def solve_disc_lyap_lrcf(A, E, B, Q=None, S=None, hess=False, trans=False, options=None,
                         default_dense_solver_backend=_DEFAULT_LYAP_SOLVER_BACKEND['disc']['dense']):
    """Compute an approximate low-rank solution of a discrete-time Lyapunov equation.

    Returns a low-rank Cholesky factor :math:`Z` such that :math:`Z Z^T` approximates the solution
    :math:`X` of a (generalized) discrete-time algebraic Lyapunov equation:

    - if trans is `False` and E is `None`:

      .. math::
         A X A^T - X + B B^T = 0,

    - if trans is `False` and E is an |Operator|:

      .. math::
          A X A^T - E X E^T + B B^T = 0,

    - if trans is `True` and E is `None`:

      .. math::
          A^T X A - X + B^T B = 0,

    - if trans is `True` and E is an |Operator|:

      .. math::
          A^T X A - E^T X E + B^T B = 0.

    - if hess is `True` and E is `None`, A is in upper Hessenberg form.

    - if hess is `True` and E is an |Operator|, (A, E) is in generalized Hessenberg form.

    We assume A and E are real |Operators|, E is invertible, and all the eigenvalues of (A, E) all
    lie inside the unit circle. Operator B needs to be given as a |VectorArray| from `A.source`, and
    for large-scale problems, we assume `len(B)` is small.

    Q and S are real |Operators|. If they are given, they contain left and right Schur vectors of (A, E)
    respectively. In this case, A contains its upper triangular matrix U and E contains its upper
    triangular matrix V of the generalized Schur decomposition of (A, E),
    where :math:`A = Q U S^T` and :math:`E = Q V S^T` .

    If Q is given and S is `None`, E is supposed to be `None` as well. In this case, A contains its
    upper triangular matrix U beeing the Schur decomposition of A, Q contains the Schur vectors of A.
    :math:`A = Q U Q^T` .

    Parameters `Q`, `S` and `hess` are supported only by pymepack solvers. When another solver is
    selected, `NotImplementedError` will be raised.

    If the solver is not specified using the options argument, a solver backend is chosen based on
    availability in the following order:

      1. `slycot` (see :func:`pymor.bindings.slycot.solve_lyap_lrcf`),
      2. `pymepack` (see :func:`pymor.bindings.pymepack.solve_lyap_lrcf`),
      3. `scipy` (see :func:`pymor.bindings.scipy.solve_lyap_lrcf`).

    Parameters
    ----------
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    Q
        The non-parametric |Operator| Q or `None`.
    S
        The non-parametric |Operator| S or `None`.
    hess
        Whether (A, E) or A is in (generalized) Hessenberg form.
    trans
        Whether the first |Operator| in the Lyapunov equation is transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.bindings.scipy.lyap_lrcf_solver_options`,
        - :func:`pymor.bindings.slycot.lyap_lrcf_solver_options`,
        - :func:`pymor.bindings.pymepack.lyap_lrcf_solver_options`.

    default_dense_solver_backend
        Default dense solver backend to use (slycot, pymepack, scipy).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Lyapunov equation solution, |VectorArray| from `A.source`.
    """
    _solve_lyap_lrcf_check_args(A, E, B, trans, Q, S)
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        backend = default_dense_solver_backend
    if backend != 'pymepack':
        if (Q is not None or S is not None) :
            raise NotImplementedError(f'Backend solver ({backend}) does not support Schur factorization.')
        if hess:
            raise NotImplementedError(f'Backend solver ({backend}) does not support Hessenberg form.')
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'pymepack':
        from pymor.bindings.pymepack import solve_lyap_lrcf as solve_lyap_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    if backend == 'pymepack':
        return solve_lyap_impl(A, E, B, Q=Q, S=S, hess=hess, trans=trans, cont_time=False, options=options)
    return solve_lyap_impl(A, E, B, trans=trans, cont_time=False, options=options)


def _solve_lyap_lrcf_check_args(A, E, B, trans, Q=None, S=None):
    assert isinstance(A, Operator) and A.linear
    assert not A.parametric
    assert A.source == A.range
    if E is not None:
        assert isinstance(E, Operator) and E.linear
        assert not E.parametric
        assert E.source == E.range
        assert E.source == A.source
    assert B in A.source
    if Q is not None:
        assert isinstance(Q, Operator) and Q.linear
    if S is not None:
        assert isinstance(S, Operator) and S.linear


@defaults('default_solver_backend')
def solve_cont_lyap_dense(A, E, B, Q=None, S=None, hess=False, trans=False, options=None,
                          default_solver_backend=_DEFAULT_LYAP_SOLVER_BACKEND['cont']['dense']):
    """Compute the solution of a continuous-time Lyapunov equation.

    Returns the solution :math:`X` of a (generalized) continuous-time algebraic Lyapunov equation:

    - if trans is `False` and E is `None`:

      .. math::
          A X + X A^T + B B^T = 0,

    - if trans is `False` and E is a |NumPy array|:

      .. math::
          A X E^T + E X A^T + B B^T = 0,

    - if trans is `True` and E is `None`:

      .. math::
          A^T X + X A + B^T B = 0,

    - if trans is `True` and E is a |NumPy array|:

      .. math::
          A^T X E + E^T X A + B^T B = 0.

    - if hess is `True` and E is `None`, A is in upper Hessenberg form.

    - if hess is `True` and E is a |NumPy array|, (A, E) is in generalized Hessenberg form.

    We assume A and E are real |NumPy arrays|, E is invertible, and that no two eigenvalues of
    (A, E) sum to zero (i.e., there exists a unique solution X).

    Q and S are real |NumPy arrays|. If they are given, they contain left and right Schur vectors of (A, E)
    respectively. In this case, A contains its upper triangular matrix U and E contains its upper
    triangular matrix V of the generalized Schur decomposition of (A, E),
    where :math:`A = Q U S^T` and :math:`E = Q V S^T` .

    If Q is given and S is `None`, E is supposed to be `None` as well. In this case, A contains its
    upper triangular matrix U beeing the Schur decomposition of A, Q contains the Schur vectors of A.
    :math:`A = Q U Q^T` .

    Parameters `Q`, `S` and `hess` are supported only by pymepack solvers. When another solver is
    selected, `NotImplementedError` will be raised.

    If the solver is not specified using the options argument, a solver backend is chosen based on
    availability in the following order:

    1. `slycot` (see :func:`pymor.bindings.slycot.solve_lyap_dense`)
    2. `scipy` (see :func:`pymor.bindings.scipy.solve_lyap_dense`)
    3. `pymepack` (see :func:`pymor.bindings.pymepack.solve_lyap_dense`)

    Parameters
    ----------
    A
        The matrix A as a 2D |NumPy array|.
    E
        The matrix E as a 2D |NumPy array| or `None`.
    B
        The matrix B as a 2D |NumPy array|.
    Q
        The matrix Q as a 2D |NumPy array| or `None`.
    S
        The matrix S as a 2D |NumPy array| or `None`.
    hess
        Whether (A, E) or A is in (generalized) Hessenberg form.
    trans
        Whether the first operator in the Lyapunov equation is transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.bindings.scipy.lyap_dense_solver_options`,
        - :func:`pymor.bindings.slycot.lyap_dense_solver_options`,
        - :func:`pymor.bindings.pymess.lyap_dense_solver_options`,
        - :func:`pymor.bindings.pymepack.lyap_dense_solver_options`.

    default_solver_backend
        Default solver backend to use (slycot, pymepack, scipy, pymess).

    Returns
    -------
    X
        Lyapunov equation solution as a |NumPy array|.
    """
    _solve_lyap_dense_check_args(A, E, B, trans)
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        backend = default_solver_backend
    if backend != 'pymepack':
        if (Q is not None or S is not None) :
            raise NotImplementedError(f'Backend solver ({backend}) does not support Schur factorization.')
        if hess:
            raise NotImplementedError(f'Backend solver ({backend}) does not support Hessenberg form.')
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_lyap_dense as solve_lyap_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_lyap_dense as solve_lyap_impl
    elif backend == 'pymess':
        from pymor.bindings.pymess import solve_lyap_dense as solve_lyap_impl
    elif backend == 'pymepack':
        from pymor.bindings.pymepack import solve_lyap_dense as solve_lyap_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    if backend == 'pymepack':
        return solve_lyap_impl(A, E, B, Q=Q, S=S, hess=hess, trans=trans, cont_time=True, options=options)
    return solve_lyap_impl(A, E, B, trans=trans, cont_time=True, options=options)


@defaults('default_solver_backend')
def solve_disc_lyap_dense(A, E, B, Q=None, S=None, hess=False, trans=False, options=None,
                          default_solver_backend=_DEFAULT_LYAP_SOLVER_BACKEND['disc']['dense']):
    """Compute the solution of a discrete-time Lyapunov equation.

    Returns the solution :math:`X` of a (generalized) continuous-time algebraic Lyapunov equation:

    - if trans is `False` and E is `None`:

      .. math::
         A X A^T - X + B B^T = 0,

    - if trans is `False` and E is a |NumPy array|:

      .. math::
          A X A^T - E X E^T + B B^T = 0,

    - if trans is `True` and E is `None`:

      .. math::
          A^T X A - X + B^T B = 0,

    - if trans is `True` and E is an |NumPy array|:

      .. math::
          A^T X A - E^T X E + B^T B = 0.

    - if hess is `True` and E is `None`, A is in upper Hessenberg form.

    - if hess is `True` and E is an |Operator|, (A, E) is in generalized Hessenberg form.

    We assume A and E are real |NumPy arrays|, E is invertible, and that all pairwise products of
    two eigenvalues of (A, E) are not equal to one (i.e., there exists a unique solution X).

    Q and S are real |NumPy arrays|. If they are given, they contain left and right Schur vectors of (A, E)
    respectively. In this case, A contains its upper triangular matrix U and E contains its upper
    triangular matrix V of the generalized Schur decomposition of (A, E),
    where :math:`A = Q U S^T` and :math:`E = Q V S^T` .

    If Q is given and S is `None`, E is supposed to be `None` as well. In this case, A contains its
    upper triangular matrix U beeing the Schur decomposition of A, Q contains the Schur vectors of A.
    :math:`A = Q S Q^T` .

    Parameters `Q`, `S` and `hess` are supported only by pymepack solvers. When another solver is
    selected, `NotImplementedError` will be raised.

    If the solver is not specified using the options argument, a solver backend is chosen based on
    availability in the following order:

    1. `slycot` (see :func:`pymor.bindings.slycot.solve_lyap_dense`)
    2. `pymepack` (see :func:`pymor.bindings.pymepack.solve_lyap_dense`)
    3. `scipy` (see :func:`pymor.bindings.scipy.solve_lyap_dense`)

    Parameters
    ----------
    A
        The matrix A as a 2D |NumPy array|.
    E
        The matrix E as a 2D |NumPy array| or `None`.
    B
        The matrix B as a 2D |NumPy array|.
    Q
        The matrix Q as a 2D |NumPy array| or `None`.
    S
        The matrix S as a 2D |NumPy array| or `None`.
    hess
        Whether (A, E) or A is in (generalized) Hessenberg form.
    trans
        Whether the first operator in the Lyapunov equation is
        transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.bindings.scipy.lyap_dense_solver_options`,
        - :func:`pymor.bindings.slycot.lyap_dense_solver_options`,
        - :func:`pymor.bindings.pymepack.lyap_dense_solver_options`.

    default_solver_backend
        Default solver backend to use (slycot, pymepack, scipy).

    Returns
    -------
    X
        Lyapunov equation solution as a |NumPy array|.
    """
    _solve_lyap_dense_check_args(A, E, B, trans, Q, S)
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        backend = default_solver_backend
    if backend != 'pymepack':
        if (Q is not None or S is not None) :
            raise NotImplementedError(f'Backend solver ({backend}) does not support Schur factorization.')
        if hess:
            raise NotImplementedError(f'Backend solver ({backend}) does not support Hessenberg form.')
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_lyap_dense as solve_lyap_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_lyap_dense as solve_lyap_impl
    elif backend == 'pymepack':
        from pymor.bindings.pymepack import solve_lyap_dense as solve_lyap_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    if backend == 'pymepack':
        return solve_lyap_impl(A, E, B, Q=Q, S=S, hess=hess, trans=trans, cont_time=False, options=options)
    return solve_lyap_impl(A, E, B, trans=trans, cont_time=False, options=options)


def _solve_lyap_dense_check_args(A, E, B, trans, Q=None, S=None):
    assert isinstance(A, np.ndarray) and A.ndim == 2
    assert A.shape[0] == A.shape[1]
    if E is not None:
        assert isinstance(E, np.ndarray) and E.ndim == 2
        assert E.shape[0] == E.shape[1]
        assert E.shape[0] == A.shape[0]
    assert isinstance(B, np.ndarray) and A.ndim == 2
    assert not trans and B.shape[0] == A.shape[0] or trans and B.shape[1] == A.shape[0]
    if Q is not None:
        assert isinstance(Q, np.ndarray) and Q.ndim == 2
        assert Q.shape[0] == Q.shape[1]
        assert Q.shape[0] == A.shape[0]
    if S is not None:
        assert E is not None and Q is not None
        assert isinstance(S, np.ndarray) and S.ndim == 2
        assert S.shape[0] == S.shape[1]
        assert S.shape[0] == A.shape[0]


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
    assert isinstance(A, np.ndarray) and A.ndim == 2
    assert A.shape[0] == A.shape[1]

    U, s, _ = spla.svd(A, lapack_driver='gesvd')
    L = U * np.sqrt(s)
    return L
