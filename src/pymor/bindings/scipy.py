# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.sparse as sps
from packaging.version import parse
from scipy.linalg import (
    lstsq,
    lu_factor,
    lu_solve,
    qr,
    solve,
    solve_continuous_are,
    solve_continuous_lyapunov,
    solve_discrete_lyapunov,
    solve_triangular,
)
from scipy.linalg.lapack import get_lapack_funcs
from scipy.sparse.linalg import LinearOperator, bicgstab, lgmres, lsqr, spilu, splu, spsolve

from pymor.core.config import config, is_scipy_mkl, is_windows_platform
from pymor.core.defaults import defaults
from pymor.core.exceptions import InversionError
from pymor.solvers.interface import Solver
from pymor.solvers.matrix.interface import (
    LyapunovSolver,
    LyapunovSolverLRCF,
    PositiveRiccatiSolver,
    PositiveRiccatiSolverLRCF,
    RiccatiSolver,
    RiccatiSolverLRCF,
)
from pymor.solvers.matrix.utils import (
    _chol,
)
from pymor.tools.weakrefcache import WeakRefCache

if config.HAVE_UMFPACK:
    import scikits.umfpack


SCIPY_1_14_OR_NEWER = parse(config.SCIPY_VERSION) >= parse('1.14')
sparray = sps.sparray if parse(config.SCIPY_VERSION) >= parse('1.11') else sps._arrays._sparray


@defaults('driver')
def svd_lapack_driver(driver='gesvd_unless_win_mkl'):
    assert driver in {'gesvd', 'gesdd', 'gesvd_unless_win_mkl'}
    if driver == 'gesvd_unless_win_mkl':
        if is_windows_platform() and is_scipy_mkl():
            from warnings import warn
            warn('Your SciPy installation seems to be using MKL as LAPACK library.\n'
                 'To avoid stability issues on Windows, we use gesdd instead of gesvd\n'
                 'for SVD computation. This may lead to reduced numerical accuracy.\n'
                 'See https://github.com/pymor/pymor/issues/2391 for further discussion.\n'
                 'To silence this warning, set the pymor.bindings.scipy.svd_lapack_driver.driver\n'
                 'default to either "gesvd" or "gesdd".')
            return 'gesdd'
        else:
            return 'gesvd'
    return driver


class ScipyLinearSolver(Solver):

    @defaults('check_finite')
    def __init__(self, check_finite=True):
        self.__auto_init(locals())

    def _solve(self, operator, V, mu, initial_guess):
        operator = operator.assemble(mu)
        from pymor.operators.numpy import NumpyMatrixOperator
        if isinstance(operator, NumpyMatrixOperator):
            matrix = operator.matrix
        else:
            from pymor.algorithms.to_matrix import to_matrix
            matrix = to_matrix(operator)
        V = V.to_numpy()
        initial_guess = initial_guess.to_numpy() if initial_guess is not None else None
        promoted_type = np.promote_types(matrix.dtype, V.dtype)

        R = self._solve_impl(matrix, V, initial_guess, promoted_type)

        if self.check_finite:
            if not np.isfinite(np.sum(R)):
                raise InversionError('Result contains non-finite values')

        return operator.source.from_numpy(R), {}

    def _solve_impl(self, matrix, V, initial_guess, promoted_type):
        raise NotImplementedError


class ScipyBicgStabSolver(ScipyLinearSolver):

    @defaults('tol', 'maxiter')
    def __init__(self, check_finite=None, tol=1e-15, maxiter=None):
        super().__init__(check_finite)
        self.__auto_init(locals())

    def _solve_impl(self, matrix, V, initial_guess, promoted_type):
        R = np.empty((matrix.shape[1], V.shape[1]), dtype=promoted_type, order='F')
        for i in range(V.shape[1]):
            if SCIPY_1_14_OR_NEWER:
                R[:, i], info = bicgstab(matrix, V[:, i], initial_guess[:, i] if initial_guess is not None else None,
                                         atol=self.tol, rtol=self.tol, maxiter=self.maxiter)
            else:
                R[:, i], info = bicgstab(matrix, V[:, i], initial_guess[:, i] if initial_guess is not None else None,
                                         tol=self.tol, maxiter=self.maxiter, atol='legacy')
            if info != 0:
                if info > 0:
                    raise InversionError(f'bicgstab failed to converge after {info} iterations')
                else:
                    raise InversionError(f'bicgstab failed with error code {info} (illegal input or breakdown)')
        return R


class ScipyBicgStabSpILUSolver(ScipyLinearSolver):

    @defaults('tol', 'maxiter', 'spilu_drop_tol', 'spilu_fill_factor', 'spilu_drop_rule', 'spilu_permc_spec')
    def __init__(self, check_finite=None, tol=1e-15, maxiter=None,
                 spilu_drop_tol=1e-4, spilu_fill_factor=10, spilu_drop_rule=None, spilu_permc_spec='COLAMD'):
        super().__init__(check_finite)
        self.__auto_init(locals())

    def _solve_impl(self, matrix, V, initial_guess, promoted_type):
        R = np.empty((matrix.shape[1], V.shape[1]), dtype=promoted_type, order='F')
        ilu = spilu(matrix, drop_tol=self.spilu_drop_tol, fill_factor=self.spilu_fill_factor,
                    drop_rule=self.spilu_drop_rule, permc_spec=self.spilu_permc_spec)
        precond = LinearOperator(matrix.shape, ilu.solve)
        for i in range(V.shape[1]):
            if SCIPY_1_14_OR_NEWER:
                R[:, i], info = bicgstab(matrix, V[:, i], initial_guess[:, i] if initial_guess is not None else None,
                                         atol=self.tol, rtol=self.tol, maxiter=self.maxiter, M=precond)
            else:
                R[:, i], info = bicgstab(matrix, V[:, i], initial_guess[:, i] if initial_guess is not None else None,
                                         tol=self.tol, maxiter=self.maxiter, M=precond, atol='legacy')
            if info != 0:
                if info > 0:
                    raise InversionError(f'bicgstab failed to converge after {info} iterations')
                else:
                    raise InversionError(f'bicgstab failed with error code {info} (illegal input or breakdown)')
        return R


class ScipySpSolveSolver(ScipyLinearSolver):

    _factorizations = WeakRefCache()

    @defaults('permc_spec', 'keep_factorization', 'use_umfpack')
    def __init__(self, check_finite=None, permc_spec='COLAMD', keep_factorization=True, use_umfpack=True):
        super().__init__(check_finite)
        self.__auto_init(locals())

    def _solve_impl(self, matrix, V, initial_guess, promoted_type):
        # convert to csc
        if not sps.isspmatrix_csc(matrix):
            # csr is also fine when using umfpack
            if sps.isspmatrix_csr(matrix) and self.use_umfpack and config.HAVE_UMFPACK:
                pass
            else:
                matrix = matrix.tocsc()

        try:
            if self.keep_factorization:
                try:
                    fac, dtype = self._factorizations.get(matrix)
                    if not np.can_cast(V.dtype, dtype, casting='safe'):
                        raise KeyError
                except KeyError:
                    matrix = matrix_astype_nocopy(matrix, promoted_type)
                    if self.use_umfpack and config.HAVE_UMFPACK:
                        fac = scikits.umfpack.splu(matrix)
                    else:
                        fac = splu(matrix, permc_spec=self.permc_spec)
                    self._factorizations.set(matrix, (fac, promoted_type))
                # we may use a complex factorization of a real matrix to
                # apply it to a real vector. In that case, we downcast
                # the result here, removing the imaginary part,
                # which should be zero.
                R = fac.solve(V).astype(promoted_type, copy=False)
            else:
                # the matrix is always converted to the promoted type.
                # if matrix.dtype == promoted_type, this is a no_op
                matrix = matrix_astype_nocopy(matrix, promoted_type)
                if self.use_umfpack and config.HAVE_UMFPACK:
                    R = scikits.umfpack.spsolve(matrix, V)
                else:
                    R = spsolve(matrix, V, permc_spec=self.permc_spec, use_umfpack=False)
            return R
        except RuntimeError as e:
            raise InversionError(e) from e

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_factorizations'] = None
        return state


class ScipyLGMRESSolver(ScipyLinearSolver):

    @defaults('tol', 'maxiter', 'inner_m', 'outer_k')
    def __init__(self, check_finite=None, tol=1e-5, maxiter=1000, inner_m=39, outer_k=3):
        super().__init__(check_finite)
        self.__auto_init(locals())

    def _solve_impl(self, matrix, V, initial_guess, promoted_type):
        R = np.empty((matrix.shape[1], V.shape[1]), dtype=promoted_type, order='F')
        for i in range(V.shape[1]):
            if SCIPY_1_14_OR_NEWER:
                R[:, i], info = lgmres(matrix, V[:, i], initial_guess[:, i] if initial_guess is not None else None,
                                       atol=self.tol,
                                       rtol=self.tol,
                                       maxiter=self.maxiter,
                                       inner_m=self.inner_m,
                                       outer_k=self.outer_k)
            else:
                R[:, i], info = lgmres(matrix, V[:, i], initial_guess[:, i] if initial_guess is not None else None,
                                       tol=self.tol,
                                       atol=self.tol,
                                       maxiter=self.maxiter,
                                       inner_m=self.inner_m,
                                       outer_k=self.outer_k)
            if info > 0:
                raise InversionError(f'lgmres failed to converge after {info} iterations')
            assert info == 0
        return R


class ScipyLSMRSolver(ScipyLinearSolver):

    least_squares = True

    @defaults('damp', 'atol', 'btol', 'conlim', 'maxiter', 'show')
    def __init__(self, check_finite=None, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8, maxiter=None, show=False):
        super().__init__(check_finite)
        self.__auto_init(locals())

    def _solve_impl(self, matrix, V, initial_guess, promoted_type):
        from scipy.sparse.linalg import lsmr
        R = np.empty((matrix.shape[1], V.shape[1]), dtype=promoted_type, order='F')
        for i in range(V.shape[1]):
            R[:, i], info, itn, _, _, _, _, _ = \
                lsmr(matrix, V[:, i],
                     damp=self.damp,
                     atol=self.atol,
                     btol=self.btol,
                     conlim=self.conlim,
                     maxiter=self.maxiter,
                     show=self.show,
                     x0=initial_guess[:, i] if initial_guess is not None else None)
            assert 0 <= info <= 7
            if info == 7:
                raise InversionError(f'lsmr failed to converge after {itn} iterations')
        return R


class ScipyLSQRSolver(ScipyLinearSolver):

    least_squares = True

    @defaults('damp', 'atol', 'btol', 'conlim', 'iter_lim', 'show')
    def __init__(self, check_finite=None, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8, iter_lim=None, show=False):
        super().__init__(check_finite)
        self.__auto_init(locals())

    def _solve_impl(self, matrix, V, initial_guess, promoted_type):
        R = np.empty((matrix.shape[1], V.shape[1]), dtype=promoted_type, order='F')
        for i in range(V.shape[1]):
            R[:, i], info, itn, _, _, _, _, _, _, _ = \
                lsqr(matrix, V[:, i],
                     damp=self.damp,
                     atol=self.atol,
                     btol=self.btol,
                     conlim=self.conlim,
                     iter_lim=self.iter_lim,
                     show=self.show,
                     x0=initial_guess[:, i] if initial_guess is not None else None)
            assert 0 <= info <= 7
            if info == 7:
                raise InversionError(f'lsmr failed to converge after {itn} iterations')
        return R


class ScipyLUSolveSolver(ScipyLinearSolver):

    _lu_factors = WeakRefCache()

    @defaults('check_cond')
    def __init__(self, check_finite=None, check_cond=True):
        super().__init__(check_finite)
        self.__auto_init(locals())

    def _solve_impl(self, matrix, V, initial_guess, promoted_type):
        try:
            lu_fac = self._lu_factors.get(matrix)
        except KeyError:
            try:
                lu_fac = lu_factor(matrix, check_finite=self.check_finite)
            except np.linalg.LinAlgError as e:
                raise InversionError(f'{type(e)!s}: {e!s}') from e
            if self.check_cond:
                gecon = get_lapack_funcs('gecon', lu_fac)
                rcond, _ = gecon(lu_fac[0], np.linalg.norm(matrix, ord=1), norm='1')
                if rcond < np.finfo(np.float64).eps:
                    self.logger.warning(f'Ill-conditioned matrix (rcond={rcond:.6g}) in solve: '
                                        'result may not be accurate.')
            self._lu_factors.set(matrix, lu_fac)
        R = lu_solve(lu_fac, V, check_finite=self.check_finite)
        return R


class ScipyLSTSQSolver(ScipyLinearSolver):

    least_squares = True

    def _solve_impl(self, matrix, V, initial_guess, promoted_type):
        try:
            R, _, _, _ = lstsq(matrix, V)
        except np.linalg.LinAlgError as e:
            raise InversionError(f'{type(e)!s}: {e!s}') from e
        return R


class ScipyQRLSTSQSolver(ScipyLinearSolver):

    least_squares = True
    _qr_factors = WeakRefCache()

    def _solve_impl(self, matrix, V, initial_guess, promoted_type):
        try:
            Q, R = self._qr_factors.get(matrix)
        except KeyError:
            try:
                Q, R = qr(matrix, mode='economic', check_finite=self.check_finite)
            except np.linalg.LinAlgError as e:
                raise InversionError(f'{type(e)!s}: {e!s}') from e
            self._qr_factors.set(matrix, (Q, R))
        U = solve_triangular(R, Q.T @ V, lower=False)
        return U


# unfortunately, this is necessary, as scipy does not
# forward the copy=False argument in its csc_matrix.astype function
def matrix_astype_nocopy(matrix, dtype):
    if matrix.dtype == dtype:
        return matrix
    else:
        return matrix.astype(dtype)


class ScipyLyapunovSolver(LyapunovSolver):
    """Compute the solution of a |LyapunovEquation|.

    This function uses `scipy.linalg.solve_continuous_lyapunov` or
    `scipy.linalg.solve_discrete_lyapunov`, which are dense solvers
    for Lyapunov equations with E=I.

    This solver has no tunable parameters.

    .. note::
        If E is not `None`, the problem will be reduced to a standard algebraic
        Lyapunov equation by inverting E.
    """

    def _solve(self, equation):
        A, E, B = equation._dense_args()
        trans = equation.trans
        cont_time = equation.cont_time

        if E is not None:
            A = solve(E, A) if not trans else solve(E.T, A.T).T
            B = solve(E, B) if not trans else solve(E.T, B.T).T
        if trans:
            A = A.T
            B = B.T
        if cont_time:
            return solve_continuous_lyapunov(A, -B @ B.T)
        else:
            return solve_discrete_lyapunov(A, B @ B.T)


class ScipyLyapunovSolverLRCF(LyapunovSolverLRCF):
    r"""Compute a low-rank Cholesky factor of a |LyapunovEquation| using SciPy.

    Computes the dense solution :math:`X` with :class:`ScipyLyapunovSolver` and
    factorizes it.  The factorization assumes :math:`X \succcurlyeq 0`, i.e. that
    the system is asymptotically stable.

    This solver has no tunable parameters.
    """

    def _solve(self, equation):
        X = ScipyLyapunovSolver()._solve(equation)
        return equation.A.source.from_numpy(_chol(X))


class ScipyRiccatiSolver(RiccatiSolver):
    """Compute the dense solution of a |RiccatiEquation| using SciPy.

    Uses `scipy.linalg.solve_continuous_are`, which is a dense solver.

    This solver has no tunable parameters.
    """

    def _solve(self, equation):
        A, E, B, C, R, S = equation._dense_args()
        trans = equation.trans

        if R is None:
            R = np.eye(C.shape[0] if not trans else B.shape[1])
        if not trans:
            if E is not None:
                E = E.T
            if S is not None:
                S = S.T
            return solve_continuous_are(A.T, C.T, B.dot(B.T), R, e=E, s=S)
        else:
            return solve_continuous_are(A, B, C.T.dot(C), R, e=E, s=S)


class ScipyRiccatiSolverLRCF(RiccatiSolverLRCF):
    r"""Compute a low-rank Cholesky factor of a |RiccatiEquation| using SciPy.

    Computes the dense solution :math:`X` with :class:`ScipyRiccatiSolver` and
    factorizes it.

    This solver has no tunable parameters.
    """

    def _solve(self, equation):
        X = ScipyRiccatiSolver()._solve(equation)
        return equation.A.source.from_numpy(_chol(X))


class ScipyPositiveRiccatiSolver(PositiveRiccatiSolver):
    """Compute the dense solution of a |PositiveRiccatiEquation| using SciPy.

    The positive Riccati equation differs from the |RiccatiEquation| only in the sign
    of the quadratic term, so it is solved by :class:`ScipyRiccatiSolver` with :math:`R`
    negated.

    This solver has no tunable parameters.
    """

    def _solve(self, equation):
        R = equation.R
        if R is None:
            R = np.eye(len(equation.C) if not equation.trans else len(equation.B))

        temp_equation = equation.with_(R=-R if R is not None else None)
        return ScipyRiccatiSolver()._solve(temp_equation)


class ScipyPositiveRiccatiSolverLRCF(PositiveRiccatiSolverLRCF):
    r"""Compute a low-rank Cholesky factor of a |PositiveRiccatiEquation| using SciPy.

    Computes the dense solution :math:`X` with :class:`ScipyPositiveRiccatiSolver` and
    factorizes it.

    This solver has no tunable parameters.
    """

    def _solve(self, equation):
        X = ScipyPositiveRiccatiSolver()._solve(equation)
        return equation.A.source.from_numpy(_chol(X))
