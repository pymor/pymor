# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import ImmutableObject
from pymor.core.config import config
from pymor.solvers.matrix.equations import LyapunovEquation, PositiveRiccatiEquation, RiccatiEquation
from pymor.solvers.matrix.utils import mat_eqn_sparse_min_size


class LyapunovSolver(ImmutableObject):
    """Compute the dense solution of a |LyapunovEquation|.

    Parameters
    ----------
    backend
        `'slycot'` or `'scipy'`, or `None` to use `slycot` if available.
    """

    def __init__(self, backend=None):
        assert backend in (None, 'slycot', 'scipy')
        self.__auto_init(locals())

    def solve(self, equation):
        r"""Solves a |LyapunovEquation|.

        A solver backend, if not provided, is chosen based in the following order:

          1. `slycot` (see :func:`pymor.bindings.slycot.solve_lyap_dense`),
          2. `scipy` (see :func:`pymor.bindings.scipy.solve_lyap_dense`).

        Parameters
        ----------
        equation
            The |LyapunovEquation| to solve.

        Returns
        -------
        X
            |LyapunovEquation| solution as a |NumPy array|.
        """
        assert isinstance(equation, LyapunovEquation)
        backend = self.backend or _dense_backend()
        _warn_dense_fallback(self, equation, backend)
        A, E, B = equation._dense_args()
        solve = _backend_module(backend).solve_lyap_dense
        return solve(A, E, B, trans=equation.trans, cont_time=equation.cont_time)


class LyapunovSolverLRCF(ImmutableObject):
    """Compute a low-rank Cholesky factor of a |LyapunovEquation|.

    Returns a low-rank Cholesky factor :math:`Z` such that :math:`Z Z^T` approximates the solution
    :math:`X` of a (generalized) continuous-time algebraic |LyapunovEquation|.

    Parameters
    ----------
    backend
        `'lradi'`, `'slycot'` or `'scipy'`, or `None` to choose automatically:
        `lradi` for large continuous-time problems, a dense backend otherwise.
        `lradi` cannot solve discrete-time equations.
    options
        Solver options forwarded to `lradi` (`tol`, `maxiter`, `shifts`,
        `shifted_system_solver`, `shift_options`), or `None`.  The dense backends
        expose no tunable options, so `options` must be `None` for those.
    """

    def __init__(self, backend=None, options=None):
        assert backend in (None, 'lradi', 'slycot', 'scipy')
        assert options is None or backend in (None, 'lradi')
        self.__auto_init(locals())

    def _auto_backend(self, equation):
        if not equation.cont_time:
            return _dense_backend()
        if equation.dim < mat_eqn_sparse_min_size():
            return _dense_backend()
        return 'lradi'

    def solve(self, equation):
        r"""Solves a |LyapunovEquation|.

        A solver backend is chosen based on availability in the following order:

        - for sparse, time-continous problems (minimum size specified by
          :func:`~pymor.solvers.matrix.utils.mat_eqn_sparse_min_size`)

          1. `lradi` (see :func:`pymor.algorithms.lradi.solve_lyap_lrcf`),

        - for dense problems (smaller than
          :func:`~pymor.solvers.matrix.utils.mat_eqn_sparse_min_size`) or time-discrete problems

          1. `slycot` (see :func:`pymor.bindings.slycot.solve_lyap_lrcf`),
          2. `scipy` (see :func:`pymor.bindings.scipy.solve_lyap_lrcf`).

        Parameters
        ----------
        equation
            The |LypaunovEquation| to solve.

        Returns
        -------
        Z
            Low-rank Cholesky factor of the Lyapunov equation solution,
            |VectorArray| from `A.source`.
        """
        assert isinstance(equation, LyapunovEquation)
        backend = self.backend or self._auto_backend(equation)
        if backend == 'lradi':
            if not equation.cont_time:
                raise ValueError('lradi solves only continuous-time Lyapunov equations.')
            options = _sparse_options(self.options, 'lradi')
        else:
            _warn_dense_fallback(self, equation, backend)
            options = None
        solve = _backend_module(backend).solve_lyap_lrcf
        return solve(equation.A, equation.E, equation.B,
                     trans=equation.trans, cont_time=equation.cont_time, options=options)


class RiccatiSolver(ImmutableObject):
    """Compute the dense solution of a |RiccatiEquation|.

    Parameters
    ----------
    backend
        `'slycot'` or `'scipy'`, or `None` to use `slycot` if available.
    """

    def __init__(self, backend=None):
        assert backend in (None, 'slycot', 'scipy')
        self.__auto_init(locals())

    def solve(self, equation):
        r"""Solves a |RiccatiEquation|.

        A solver backend, if not provided, is chosen based in the following order:

          1. `slycot` (see :func:`pymor.bindings.slycot.solve_ricc_dense`),
          2. `scipy` (see :func:`pymor.bindings.scipy.solve_ricc_dense`).

        Parameters
        ----------
        equation
            The |RiccatiEquation| to solve.

        Returns
        -------
        X
            |RiccatiEquation| solution as a |NumPy array|.
        """
        assert isinstance(equation, RiccatiEquation)
        backend = self.backend or _dense_backend()
        _warn_dense_fallback(self, equation, backend)
        A, E, B, C, R, S = equation._dense_args()
        solve = _backend_module(backend).solve_ricc_dense
        return solve(A, E, B, C, R=R, S=S, trans=equation.trans)


class RiccatiSolverLRCF(ImmutableObject):
    """Compute a low-rank Cholesky factor of a |RiccatiEquation|.

    Parameters
    ----------
    backend
        `'lrradi'`, `'slycot'` or `'scipy'`, or `None` to choose automatically:
        `lrradi` for large problems, a dense backend otherwise.
    options
        Solver options forwarded to `lrradi` (`tol`, `maxiter`, `shifts`,
        `shifted_system_solver`, `shift_options`), or `None`.  Must be `None` for
        the dense backends.
    """

    def __init__(self, backend=None, options=None):
        assert backend in (None, 'lrradi', 'slycot', 'scipy')
        assert options is None or backend in (None, 'lrradi')
        self.__auto_init(locals())

    def _auto_backend(self, equation):
        return 'lrradi' if equation.dim >= mat_eqn_sparse_min_size() else _dense_backend()

    def solve(self, equation):
        r"""Solves a |RiccatiEquation|.

        A solver backend, if not provided, is chosen based in the following order:

        - for sparse problems (minimum size specified by
          :func:`~pymor.matrix.solvers.utils.mat_eqn_sparse_min_size`)

          1. `lrradi` (see :func:`pymor.algorithms.lrradi.solve_ricc_lrcf`),

        - for dense problems (smaller than
          :func:`~pymor.solvers.matrix.utils.mat_eqn_sparse_min_size`)

          1. `slycot` (see :func:`pymor.bindings.slycot.solve_ricc_lrcf`),
          2. `scipy` (see :func:`pymor.bindings.scipy.solve_ricc_lrcf`).

        Parameters
        ----------
        equation
            The |RiccatiEquation| to solve.

        Returns
        -------
        Z
            Low-rank Cholesky factor of the |RiccatiEquation| solution,
            |VectorArray| from `A.source`.
        """
        assert isinstance(equation, RiccatiEquation)
        backend = self.backend or self._auto_backend(equation)
        if backend == 'lrradi':
            options = _sparse_options(self.options, 'lrradi')
        else:
            _warn_dense_fallback(self, equation, backend)
            options = None
        solve = _backend_module(backend).solve_ricc_lrcf
        return solve(equation.A, equation.E, equation.B, equation.C,
                     R=equation.R, S=equation.S, trans=equation.trans, options=options)


class PositiveRiccatiSolver(ImmutableObject):
    """Compute the dense solution of a |PositiveRiccatiEquation|.

    Parameters
    ----------
    backend
        `'slycot'` or `'scipy'`, or `None` to use `slycot` if available.
    """

    def __init__(self, backend=None):
        assert backend in (None, 'slycot', 'scipy')
        self.__auto_init(locals())

    def solve(self, equation):
        r"""Solves a |PositiveRiccatiEquation|.

        A solver backend, if not provided, is chosen based in the following order:

          1. `slycot` (see :func:`pymor.bindings.slycot.solve_pos_ricc_dense`),
          2. `scipy` (see :func:`pymor.bindings.scipy.solve_pos_ricc_dense`).

        Parameters
        ----------
        equation
            The |PositiveRiccatiEquation| to solve.

        Returns
        -------
        X
            |PositiveRiccatiEquation| solution as a |NumPy array|.
        """
        assert isinstance(equation, PositiveRiccatiEquation)
        backend = self.backend or _dense_backend()
        _warn_dense_fallback(self, equation, backend)
        A, E, B, C, R, S = equation._dense_args()
        solve = _backend_module(backend).solve_pos_ricc_dense
        return solve(A, E, B, C, R=R, S=S, trans=equation.trans)


class PositiveRiccatiSolverLRCF(ImmutableObject):
    """Compute a low-rank Cholesky factor of a |PositiveRiccatiEquation|.

    Parameters
    ----------
    backend
        `'slycot'` or `'scipy'`, or `None` to use `slycot` if available.
    """

    def __init__(self, backend=None):
        assert backend in (None, 'slycot', 'scipy')
        self.__auto_init(locals())

    def solve(self, equation):
        r"""Solves a |PositiveRiccatiEquation|.

        A solver backend, if not provided, is chosen based in the following order:

          1. `slycot` (see :func:`pymor.bindings.slycot.solve_pos_ricc_lrcf`),
          2. `scipy` (see :func:`pymor.bindings.scipy.solve_pos_ricc_lrcf`).

        Currently, only dense solvers are provided.

        Parameters
        ----------
        equation
            The |PositiveRiccatiEquation| to solve.

        Returns
        -------
        Z
            Low-rank Cholesky factor of the |PositiveRiccatiEquation| solution,
            |VectorArray| from `A.source`.
        """
        assert isinstance(equation, PositiveRiccatiEquation)
        backend = self.backend or _dense_backend()
        _warn_dense_fallback(self, equation, backend)
        solve = _backend_module(backend).solve_pos_ricc_lrcf
        return solve(equation.A, equation.E, equation.B, equation.C,
                     R=equation.R, S=equation.S, trans=equation.trans)

def _dense_backend():
    return 'slycot' if config.HAVE_SLYCOT else 'scipy'


def _backend_module(backend):
    if backend == 'scipy':
        from pymor.bindings import scipy as backend_module
    elif backend == 'slycot':
        from pymor.bindings import slycot as backend_module
    elif backend == 'lradi':
        from pymor.algorithms import lradi as backend_module
    elif backend == 'lrradi':
        from pymor.algorithms import lrradi as backend_module
    else:
        raise ValueError(f'Unknown backend ({backend}).')
    return backend_module


def _sparse_options(options, type_):
    """Hand `options` to `lradi`/`lrradi`.

    `_parse_options` asserts `'type' in options`, which is an implementation
    detail of the backend dispatch inside `algorithms/lyapunov.py`.  Since the
    backend is already fixed by the solver, inject it here so that callers can
    pass a plain `{'tol': 1e-8, 'maxiter': 200}`.
    """
    if options is None:
        return None
    options = dict(options)
    options['type'] = type_
    return options


def _warn_dense_fallback(solver, equation, backend):
    if equation.dim >= mat_eqn_sparse_min_size():
        solver.logger.warning(
            f'Using the dense {backend} backend on a {equation.dim} x {equation.dim} problem; '
            'this may be expensive in time and memory.')


class MatrixSolvers(ImmutableObject):
    """Bundle of the six matrix-equation solvers used by a |Model|.

    Parameters
    ----------
    lyapunov
        A :class:`LyapunovSolver` or `None`.
    lyapunov_lrcf
        A :class:`LyapunovSolverLRCF` or `None`.
    riccati
        A :class:`RiccatiSolver` or `None`.
    riccati_lrcf
        A :class:`RiccatiSolverLRCF` or `None`.
    positive_riccati
        A :class:`PositiveRiccatiSolver` or `None`.
    positive_riccati_lrcf
        A :class:`PositiveRiccatiSolverLRCF` or `None`.
    """

    def __init__(self, lyapunov=None, lyapunov_lrcf=None,
                 riccati=None, riccati_lrcf=None,
                 positive_riccati=None, positive_riccati_lrcf=None):
        lyapunov = lyapunov or LyapunovSolver()
        lyapunov_lrcf = lyapunov_lrcf or LyapunovSolverLRCF()
        riccati = riccati or RiccatiSolver()
        riccati_lrcf = riccati_lrcf or RiccatiSolverLRCF()
        positive_riccati = positive_riccati or PositiveRiccatiSolver()
        positive_riccati_lrcf = positive_riccati_lrcf or PositiveRiccatiSolverLRCF()

        assert isinstance(lyapunov, LyapunovSolver)
        assert isinstance(lyapunov_lrcf, LyapunovSolverLRCF)
        assert isinstance(riccati, RiccatiSolver)
        assert isinstance(riccati_lrcf, RiccatiSolverLRCF)
        assert isinstance(positive_riccati, PositiveRiccatiSolver)
        assert isinstance(positive_riccati_lrcf, PositiveRiccatiSolverLRCF)

        self.__auto_init(locals())
