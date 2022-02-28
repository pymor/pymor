# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import numpy as np
from scipy.linalg import solve, solve_continuous_lyapunov, solve_discrete_lyapunov, solve_continuous_are
from scipy.sparse.linalg import bicgstab, spsolve, splu, spilu, lgmres, lsqr, LinearOperator

from pymor.algorithms.lyapunov import _solve_lyap_lrcf_check_args, _solve_lyap_dense_check_args, _chol
from pymor.algorithms.riccati import _solve_ricc_check_args, _solve_ricc_dense_check_args
from pymor.algorithms.genericsolvers import _parse_options
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.core.exceptions import InversionError
from pymor.operators.numpy import NumpyMatrixOperator


@defaults('bicgstab_tol', 'bicgstab_maxiter', 'spilu_drop_tol',
          'spilu_fill_factor', 'spilu_drop_rule', 'spilu_permc_spec', 'spsolve_permc_spec',
          'spsolve_keep_factorization',
          'lgmres_tol', 'lgmres_maxiter', 'lgmres_inner_m', 'lgmres_outer_k', 'least_squares_lsmr_damp',
          'least_squares_lsmr_atol', 'least_squares_lsmr_btol', 'least_squares_lsmr_conlim',
          'least_squares_lsmr_maxiter', 'least_squares_lsmr_show', 'least_squares_lsqr_atol',
          'least_squares_lsqr_btol', 'least_squares_lsqr_conlim', 'least_squares_lsqr_iter_lim',
          'least_squares_lsqr_show')
def solver_options(bicgstab_tol=1e-15,
                   bicgstab_maxiter=None,
                   spilu_drop_tol=1e-4,
                   spilu_fill_factor=10,
                   spilu_drop_rule=None,
                   spilu_permc_spec='COLAMD',
                   spsolve_permc_spec='COLAMD',
                   spsolve_keep_factorization=True,
                   lgmres_tol=1e-5,
                   lgmres_maxiter=1000,
                   lgmres_inner_m=39,
                   lgmres_outer_k=3,
                   least_squares_lsmr_damp=0.0,
                   least_squares_lsmr_atol=1e-6,
                   least_squares_lsmr_btol=1e-6,
                   least_squares_lsmr_conlim=1e8,
                   least_squares_lsmr_maxiter=None,
                   least_squares_lsmr_show=False,
                   least_squares_lsqr_damp=0.0,
                   least_squares_lsqr_atol=1e-6,
                   least_squares_lsqr_btol=1e-6,
                   least_squares_lsqr_conlim=1e8,
                   least_squares_lsqr_iter_lim=None,
                   least_squares_lsqr_show=False):
    """Returns available solvers with default |solver_options| for the SciPy backend.

    Parameters
    ----------
    bicgstab_tol
        See :func:`scipy.sparse.linalg.bicgstab`.
    bicgstab_maxiter
        See :func:`scipy.sparse.linalg.bicgstab`.
    spilu_drop_tol
        See :func:`scipy.sparse.linalg.spilu`.
    spilu_fill_factor
        See :func:`scipy.sparse.linalg.spilu`.
    spilu_drop_rule
        See :func:`scipy.sparse.linalg.spilu`.
    spilu_permc_spec
        See :func:`scipy.sparse.linalg.spilu`.
    spsolve_permc_spec
        See :func:`scipy.sparse.linalg.spsolve`.
    spsolve_keep_factorization
        See :func:`scipy.sparse.linalg.spsolve`.
    lgmres_tol
        See :func:`scipy.sparse.linalg.lgmres`.
    lgmres_maxiter
        See :func:`scipy.sparse.linalg.lgmres`.
    lgmres_inner_m
        See :func:`scipy.sparse.linalg.lgmres`.
    lgmres_outer_k
        See :func:`scipy.sparse.linalg.lgmres`.
    least_squares_lsmr_damp
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_atol
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_btol
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_conlim
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_maxiter
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_show
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsqr_damp
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_atol
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_btol
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_conlim
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_iter_lim
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_show
        See :func:`scipy.sparse.linalg.lsqr`.

    Returns
    -------
    A dict of available solvers with default |solver_options|.
    """
    opts = {'scipy_bicgstab_spilu':     {'type': 'scipy_bicgstab_spilu',
                                         'tol': bicgstab_tol,
                                         'maxiter': bicgstab_maxiter,
                                         'spilu_drop_tol': spilu_drop_tol,
                                         'spilu_fill_factor': spilu_fill_factor,
                                         'spilu_drop_rule': spilu_drop_rule,
                                         'spilu_permc_spec': spilu_permc_spec},
            'scipy_bicgstab':           {'type': 'scipy_bicgstab',
                                         'tol': bicgstab_tol,
                                         'maxiter': bicgstab_maxiter},
            'scipy_spsolve':            {'type': 'scipy_spsolve',
                                         'permc_spec': spsolve_permc_spec,
                                         'keep_factorization': spsolve_keep_factorization},
            'scipy_lgmres':             {'type': 'scipy_lgmres',
                                         'tol': lgmres_tol,
                                         'maxiter': lgmres_maxiter,
                                         'inner_m': lgmres_inner_m,
                                         'outer_k': lgmres_outer_k},
            'scipy_least_squares_lsqr': {'type': 'scipy_least_squares_lsqr',
                                         'damp': least_squares_lsqr_damp,
                                         'atol': least_squares_lsqr_atol,
                                         'btol': least_squares_lsqr_btol,
                                         'conlim': least_squares_lsqr_conlim,
                                         'iter_lim': least_squares_lsqr_iter_lim,
                                         'show': least_squares_lsqr_show}}

    if config.HAVE_SCIPY_LSMR:
        opts['scipy_least_squares_lsmr'] = {'type': 'scipy_least_squares_lsmr',
                                            'damp': least_squares_lsmr_damp,
                                            'atol': least_squares_lsmr_atol,
                                            'btol': least_squares_lsmr_btol,
                                            'conlim': least_squares_lsmr_conlim,
                                            'maxiter': least_squares_lsmr_maxiter,
                                            'show': least_squares_lsmr_show}

    return opts


@defaults('check_finite', 'default_solver', 'default_least_squares_solver')
def apply_inverse(op, V, initial_guess=None, options=None, least_squares=False, check_finite=True,
                  default_solver='scipy_spsolve', default_least_squares_solver='scipy_least_squares_lsmr'):
    """Solve linear equation system.

    Applies the inverse of `op` to the vectors in `V` using SciPy.

    Parameters
    ----------
    op
        The linear, non-parametric |Operator| to invert.
    V
        |VectorArray| of right-hand sides for the equation system.
    initial_guess
        |VectorArray| with the same length as `V` containing initial guesses
        for the solution.  Some implementations of `apply_inverse` may
        ignore this parameter.  If `None` a solver-dependent default is used.
    options
        The |solver_options| to use (see :func:`solver_options`).
    least_squares
        If `True`, return least squares solution.
    check_finite
        Test if solution only contains finite values.
    default_solver
        Default solver to use (scipy_spsolve, scipy_bicgstab, scipy_bicgstab_spilu,
        scipy_lgmres, scipy_least_squares_lsmr, scipy_least_squares_lsqr).
    default_least_squares_solver
        Default solver to use for least squares problems (scipy_least_squares_lsmr,
        scipy_least_squares_lsqr).

    Returns
    -------
    |VectorArray| of the solution vectors.
    """
    assert V in op.range
    assert initial_guess is None or initial_guess in op.source and len(initial_guess) == len(V)

    if isinstance(op, NumpyMatrixOperator):
        matrix = op.matrix
    else:
        from pymor.algorithms.to_matrix import to_matrix
        matrix = to_matrix(op)

    options = _parse_options(options, solver_options(), default_solver, default_least_squares_solver, least_squares)

    V = V.to_numpy()
    initial_guess = initial_guess.to_numpy() if initial_guess is not None else None
    promoted_type = np.promote_types(matrix.dtype, V.dtype)
    R = np.empty((len(V), matrix.shape[1]), dtype=promoted_type)

    if options['type'] == 'scipy_bicgstab':
        for i, VV in enumerate(V):
            R[i], info = bicgstab(matrix, VV, initial_guess[i] if initial_guess is not None else None,
                                  tol=options['tol'], maxiter=options['maxiter'], atol='legacy')
            if info != 0:
                if info > 0:
                    raise InversionError(f'bicgstab failed to converge after {info} iterations')
                else:
                    raise InversionError('bicgstab failed with error code {} (illegal input or breakdown)'.
                                         format(info))
    elif options['type'] == 'scipy_bicgstab_spilu':
        ilu = spilu(matrix, drop_tol=options['spilu_drop_tol'], fill_factor=options['spilu_fill_factor'],
                    drop_rule=options['spilu_drop_rule'], permc_spec=options['spilu_permc_spec'])
        precond = LinearOperator(matrix.shape, ilu.solve)
        for i, VV in enumerate(V):
            R[i], info = bicgstab(matrix, VV, initial_guess[i] if initial_guess is not None else None,
                                  tol=options['tol'], maxiter=options['maxiter'], M=precond, atol='legacy')
            if info != 0:
                if info > 0:
                    raise InversionError(f'bicgstab failed to converge after {info} iterations')
                else:
                    raise InversionError('bicgstab failed with error code {} (illegal input or breakdown)'.
                                         format(info))
    elif options['type'] == 'scipy_spsolve':
        try:
            # maybe remove unusable factorization:
            if hasattr(matrix, 'factorization'):
                fdtype = matrix.factorizationdtype
                if not np.can_cast(V.dtype, fdtype, casting='safe'):
                    del matrix.factorization

            if hasattr(matrix, 'factorization'):
                # we may use a complex factorization of a real matrix to
                # apply it to a real vector. In that case, we downcast
                # the result here, removing the imaginary part,
                # which should be zero.
                R = matrix.factorization.solve(V.T).T.astype(promoted_type, copy=False)
            elif options['keep_factorization']:
                # the matrix is always converted to the promoted type.
                # if matrix.dtype == promoted_type, this is a no_op
                matrix.factorization = splu(matrix_astype_nocopy(matrix.tocsc(), promoted_type),
                                            permc_spec=options['permc_spec'])
                matrix.factorizationdtype = promoted_type
                R = matrix.factorization.solve(V.T).T
            else:
                # the matrix is always converted to the promoted type.
                # if matrix.dtype == promoted_type, this is a no_op
                R = spsolve(matrix_astype_nocopy(matrix, promoted_type), V.T, permc_spec=options['permc_spec']).T
        except RuntimeError as e:
            raise InversionError(e) from e
    elif options['type'] == 'scipy_lgmres':
        for i, VV in enumerate(V):
            R[i], info = lgmres(matrix, VV, initial_guess[i] if initial_guess is not None else None,
                                tol=options['tol'],
                                atol=options['tol'],
                                maxiter=options['maxiter'],
                                inner_m=options['inner_m'],
                                outer_k=options['outer_k'])
            if info > 0:
                raise InversionError(f'lgmres failed to converge after {info} iterations')
            assert info == 0
    elif options['type'] == 'scipy_least_squares_lsmr':
        from scipy.sparse.linalg import lsmr
        for i, VV in enumerate(V):
            R[i], info, itn, _, _, _, _, _ = lsmr(matrix, VV,
                                                  damp=options['damp'],
                                                  atol=options['atol'],
                                                  btol=options['btol'],
                                                  conlim=options['conlim'],
                                                  maxiter=options['maxiter'],
                                                  show=options['show'],
                                                  x0=initial_guess[i] if initial_guess is not None else None)
            assert 0 <= info <= 7
            if info == 7:
                raise InversionError(f'lsmr failed to converge after {itn} iterations')
    elif options['type'] == 'scipy_least_squares_lsqr':
        for i, VV in enumerate(V):
            R[i], info, itn, _, _, _, _, _, _, _ = lsqr(matrix, VV,
                                                        damp=options['damp'],
                                                        atol=options['atol'],
                                                        btol=options['btol'],
                                                        conlim=options['conlim'],
                                                        iter_lim=options['iter_lim'],
                                                        show=options['show'],
                                                        x0=initial_guess[i] if initial_guess is not None else None)
            assert 0 <= info <= 7
            if info == 7:
                raise InversionError(f'lsmr failed to converge after {itn} iterations')
    else:
        raise ValueError('Unknown solver type')

    if check_finite:
        if not np.isfinite(np.sum(R)):
            raise InversionError('Result contains non-finite values')

    return op.source.from_numpy(R)


# unfortunately, this is necessary, as scipy does not
# forward the copy=False argument in its csc_matrix.astype function
def matrix_astype_nocopy(matrix, dtype):
    if matrix.dtype == dtype:
        return matrix
    else:
        return matrix.astype(dtype)


def lyap_lrcf_solver_options():
    """Return available Lyapunov solvers with default options for the SciPy backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'scipy': {'type': 'scipy'}}


def solve_lyap_lrcf(A, E, B, trans=False, cont_time=True, options=None):
    """Compute an approximate low-rank solution of a Lyapunov equation.

    See

    - :func:`pymor.algorithms.lyapunov.solve_cont_lyap_lrcf`
    - :func:`pymor.algorithms.lyapunov.solve_disc_lyap_lrcf`

    for a general description.

    This function uses `scipy.linalg.solve_continuous_lyapunov` or
    `scipy.linalg.solve_discrete_lyapunov`, which are dense solvers for Lyapunov equations with E=I.
    Therefore, we assume A and E can be converted to |NumPy arrays| using
    :func:`~pymor.algorithms.to_matrix.to_matrix` and that `B.to_numpy` is implemented.

    .. note::
        If E is not `None`, the problem will be reduced to a standard algebraic
        Lyapunov equation by inverting E.

    Parameters
    ----------
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    trans
        Whether the first |Operator| in the Lyapunov equation is transposed.
    cont_time
        Whether the continuous- or discrete-time Lyapunov equation is solved.
    options
        The solver options to use (see :func:`lyap_lrcf_solver_options`).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Lyapunov equation solution, |VectorArray| from `A.source`.
    """
    _solve_lyap_lrcf_check_args(A, E, B, trans)
    options = _parse_options(options, lyap_lrcf_solver_options(), 'scipy', None, False)

    X = solve_lyap_dense(to_matrix(A, format='dense'),
                         to_matrix(E, format='dense') if E else None,
                         B.to_numpy().T if not trans else B.to_numpy(),
                         trans=trans, cont_time=cont_time, options=options)
    return A.source.from_numpy(_chol(X).T)


def lyap_dense_solver_options():
    """Return available dense Lyapunov solvers with default options for the SciPy backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'scipy': {'type': 'scipy'}}


def solve_lyap_dense(A, E, B, trans=False, cont_time=True, options=None):
    """Compute the solution of a Lyapunov equation.

    See

    - :func:`pymor.algorithms.lyapunov.solve_cont_lyap_dense`
    - :func:`pymor.algorithms.lyapunov.solve_disc_lyap_dense`

    for a general description.

    This function uses `scipy.linalg.solve_continuous_lyapunov` or
    `scipy.linalg.solve_discrete_lyapunov`, which are dense solvers for Lyapunov equations with E=I.

    .. note::
        If E is not `None`, the problem will be reduced to a standard algebraic
        Lyapunov equation by inverting E.

    Parameters
    ----------
    A
        The matrix A as a 2D |NumPy array|.
    E
        The matrix E as a 2D |NumPy array| or `None`.
    B
        The matrix B as a 2D |NumPy array|.
    trans
        Whether the first operator in the Lyapunov equation is transposed.
    cont_time
        Whether the continuous- or discrete-time Lyapunov equation is solved.
    options
        The solver options to use (see :func:`lyap_dense_solver_options`).

    Returns
    -------
    X
        Lyapunov equation solution as a |NumPy array|.
    """
    _solve_lyap_dense_check_args(A, E, B, trans)
    options = _parse_options(options, lyap_dense_solver_options(), 'scipy', None, False)

    if options['type'] == 'scipy':
        if E is not None:
            A = solve(E, A) if not trans else solve(E.T, A.T).T
            B = solve(E, B) if not trans else solve(E.T, B.T).T
        if trans:
            A = A.T
            B = B.T
        if cont_time:
            X = solve_continuous_lyapunov(A, -B @ B.T)
        else:
            X = solve_discrete_lyapunov(A, B @ B.T)
    else:
        raise ValueError(f"Unexpected Lyapunov equation solver ({options['type']}).")

    return X


def ricc_lrcf_solver_options():
    """Return available Riccati solvers with default options for the SciPy backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'scipy': {'type': 'scipy'}}


def solve_ricc_lrcf(A, E, B, C, R=None, trans=False, options=None):
    """Compute an approximate low-rank solution of a Riccati equation.

    See :func:`pymor.algorithms.riccati.solve_ricc_lrcf` for a general
    description.

    This function uses `scipy.linalg.solve_continuous_are`, which
    is a dense solver.
    Therefore, we assume all |Operators| and |VectorArrays| can be
    converted to |NumPy arrays| using
    :func:`~pymor.algorithms.to_matrix.to_matrix` and
    :func:`~pymor.vectorarrays.interface.VectorArray.to_numpy`.

    Parameters
    ----------
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    C
        The operator C as a |VectorArray| from `A.source`.
    R
        The matrix R as a 2D |NumPy array| or `None`.
    trans
        Whether the first |Operator| in the Riccati equation is
        transposed.
    options
        The solver options to use (see :func:`ricc_lrcf_solver_options`).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Riccati equation solution,
        |VectorArray| from `A.source`.
    """
    _solve_ricc_check_args(A, E, B, C, R, trans)
    options = _parse_options(options, ricc_lrcf_solver_options(), 'scipy', None, False)
    if options['type'] != 'scipy':
        raise ValueError(f"Unexpected Riccati equation solver ({options['type']}).")

    A_source = A.source
    A = to_matrix(A, format='dense')
    E = to_matrix(E, format='dense') if E else None
    B = B.to_numpy().T
    C = C.to_numpy()
    X = solve_ricc_dense(A, E, B, C, R, trans, options)

    return A_source.from_numpy(_chol(X).T)


def ricc_dense_solver_options():
    """Return available Riccati solvers with default options for the SciPy backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'scipy': {'type': 'scipy'}}


def solve_ricc_dense(A, E, B, C, R=None, trans=False, options=None):
    """Compute the solution of a Riccati equation.

    See :func:`pymor.algorithms.riccati.solve_ricc_dense` for a general
    description.

    This function uses `scipy.linalg.solve_continuous_are`, which
    is a dense solver.

    Parameters
    ----------
    A
        The matrix A as a 2D |NumPy array|.
    E
        The matrix E as a 2D |NumPy array| or `None`.
    B
        The matrix B as a 2D |NumPy array|.
    C
        The matrix C as a 2D |NumPy array|.
    R
        The matrix R as a 2D |NumPy array| or `None`.
    trans
        Whether the first operator in the Riccati equation is
        transposed.
    options
        The solver options to use (see
        :func:`ricc_dense_solver_options`).

    Returns
    -------
    X
        Riccati equation solution as a |NumPy array|.
    """
    _solve_ricc_dense_check_args(A, E, B, C, R, trans)
    options = _parse_options(options, ricc_dense_solver_options(), 'scipy', None, False)
    if options['type'] != 'scipy':
        raise ValueError(f"Unexpected Riccati equation solver ({options['type']}).")

    if R is None:
        R = np.eye(C.shape[0] if not trans else B.shape[1])
    if not trans:
        if E is not None:
            E = E.T
        X = solve_continuous_are(A.T, C.T, B.dot(B.T), R, E)
    else:
        X = solve_continuous_are(A, B, C.T.dot(C), R, E)

    return X


def pos_ricc_lrcf_solver_options():
    """Return available positive Riccati solvers with default options for the SciPy backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'scipy': {'type': 'scipy'}}


def solve_pos_ricc_lrcf(A, E, B, C, R=None, trans=False, options=None):
    """Compute an approximate low-rank solution of a positive Riccati equation.

    See :func:`pymor.algorithms.riccati.solve_pos_ricc_lrcf` for a
    general description.

    This function uses `scipy.linalg.solve_continuous_are`, which
    is a dense solver.
    Therefore, we assume all |Operators| and |VectorArrays| can be
    converted to |NumPy arrays| using
    :func:`~pymor.algorithms.to_matrix.to_matrix` and
    :func:`~pymor.vectorarrays.interface.VectorArray.to_numpy`.

    Parameters
    ----------
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    C
        The operator C as a |VectorArray| from `A.source`.
    R
        The matrix R as a 2D |NumPy array| or `None`.
    trans
        Whether the first |Operator| in the positive Riccati equation is
        transposed.
    options
        The solver options to use (see
        :func:`pos_ricc_lrcf_solver_options`).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the positive Riccati equation
        solution, |VectorArray| from `A.source`.
    """
    _solve_ricc_check_args(A, E, B, C, R, trans)
    options = _parse_options(options, pos_ricc_lrcf_solver_options(), 'scipy', None, False)
    if options['type'] != 'scipy':
        raise ValueError(f"Unexpected positive Riccati equation solver ({options['type']}).")

    if R is None:
        R = np.eye(len(C) if not trans else len(B))
    return solve_ricc_lrcf(A, E, B, C, -R, trans, options)
