# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


from pymor.core.base import ImmutableObject
from pymor.core.config import config
from pymor.solvers.matrix.interface import (
    LyapunovSolver,
    LyapunovSolverLRCF,
    PositiveRiccatiSolver,
    PositiveRiccatiSolverLRCF,
    RiccatiSolver,
    RiccatiSolverLRCF,
)
from pymor.solvers.matrix.utils import mat_eqn_sparse_min_size


class DefaultLyapunovSolver(LyapunovSolver):
    """Default |LyapunovSolver|.

    A solver backend is chosen based on availability in the following order:

    1. `slycot` (:class:`~pymor.bindings.slycot.SlycotLyapunovSolver`),
    2. `scipy` (:class:`~pymor.bindings.scipy.ScipyLyapunovSolver`).
    """

    def _solve(self, equation):
        if _dense_backend() == 'slycot':
            from pymor.bindings.slycot import SlycotLyapunovSolver
            solver = SlycotLyapunovSolver()
        else:
            from pymor.bindings.scipy import ScipyLyapunovSolver
            solver = ScipyLyapunovSolver()
        return solver.solve(equation)


class DefaultLyapunovSolverLRCF(LyapunovSolverLRCF):
    """Default |LyapunovSolverLRCF|.

    A solver backend is chosen based on availability in the following order:

        - for sparse, continous-time problems (minimum size specified by
          :func:`~pymor.solvers.matrix.utils.mat_eqn_sparse_min_size`)

          1. `lradi` (see :func:`pymor.solvers.matrix.lradi.LradiLyapunovSolverLRCF`),

        - for dense problems (smaller than
          :func:`~pymor.solvers.matrix.utils.mat_eqn_sparse_min_size`) or discrete-time problems

          1. `slycot` (see :class:`pymor.bindings.slycot.SlycotLyapunovSolverLRCF`),
          2. `scipy` (see :class:`pymor.bindings.scipy.ScipyLyapunovSolverLRCF`).
    """

    @staticmethod
    def _auto_backend(equation):
        if not equation.cont_time:
            return _dense_backend()
        if equation.dim < mat_eqn_sparse_min_size():
            return _dense_backend()
        return 'lradi'

    def _solve(self, equation):
        backend = self._auto_backend(equation)
        if backend == 'lradi':
            if not equation.cont_time:
                raise ValueError('lradi solves only continuous-time Lyapunov equations.')
            from pymor.solvers.matrix.lradi import LradiLyapunovSolverLRCF
            solver = LradiLyapunovSolverLRCF()
        else:
            _warn_dense_fallback(self, equation, backend)
            if _dense_backend() == 'slycot':
                from pymor.bindings.slycot import SlycotLyapunovSolverLRCF
                solver = SlycotLyapunovSolverLRCF()
            else:
                from pymor.bindings.scipy import ScipyLyapunovSolverLRCF
                solver = ScipyLyapunovSolverLRCF()

        return solver.solve(equation)

class DefaultRiccatiSolver(RiccatiSolver):
    r"""Default |RiccatiSolver|.

    A solver backend is chosen based in the following order:

          1. `slycot` (see :class:`pymor.bindings.slycot.SlycotRiccatiSolver`),
          2. `scipy` (see :class:`pymor.bindings.scipy.ScipyRiccatiSolver`).
    """

    def _solve(self, equation):
        if _dense_backend() == 'slycot':
            from pymor.bindings.slycot import SlycotRiccatiSolver
            solver = SlycotRiccatiSolver()
        else:
            from pymor.bindings.scipy import ScipyRiccatiSolver
            solver = ScipyRiccatiSolver()
        return solver.solve(equation)


class DefaultRiccatiSolverLRCF(RiccatiSolverLRCF):
    r"""Default |RiccatiSolverLRCF|.

    A solver backend, if not provided, is chosen based in the following order:

        - for sparse problems (minimum size specified by
          :func:`~pymor.solvers.matrix.utils.mat_eqn_sparse_min_size`)

          1. `lrradi` (see :class:`pymor.solvers.matrix.lrradi.LrradiRiccatiSolverLRCF`),

        - for dense problems (smaller than
          :func:`~pymor.solvers.matrix.utils.mat_eqn_sparse_min_size`)

          1. `slycot` (see :class:`pymor.bindings.slycot.SlycotRiccatiSolverLRCF`),
          2. `scipy` (see :class:`pymor.bindings.scipy.ScipyRiccatiSolverLRCF`).
    """

    @staticmethod
    def _auto_backend(equation):
        return 'lrradi' if equation.dim >= mat_eqn_sparse_min_size() else _dense_backend()

    def _solve(self, equation):
        backend = self._auto_backend(equation)
        if backend == 'lrradi':
            from pymor.solvers.matrix.lrradi import LrradiRiccatiSolverLRCF
            solver = LrradiRiccatiSolverLRCF()
        else:
            _warn_dense_fallback(self, equation, backend)
            if _dense_backend() == 'slycot':
                from pymor.bindings.slycot import SlycotRiccatiSolverLRCF
                solver = SlycotRiccatiSolverLRCF()
            else:
                from pymor.bindings.scipy import ScipyRiccatiSolverLRCF
                solver = ScipyRiccatiSolverLRCF()

        return solver.solve(equation)


class DefaultPositiveRiccatiSolver(PositiveRiccatiSolver):
    r"""Default |PositiveRiccatiSolver|.

    A solver backend is chosen based in the following order:

          1. `slycot` (see :class:`pymor.bindings.slycot.SlycotPositiveRiccatiSolver`),
          2. `scipy` (see :class:`pymor.bindings.scipy.ScipyPositiveRiccatiSolver`).
    """

    def _solve(self, equation):
        if _dense_backend() == 'slycot':
            from pymor.bindings.slycot import SlycotPositiveRiccatiSolver
            solver = SlycotPositiveRiccatiSolver()
        else:
            from pymor.bindings.scipy import ScipyPositiveRiccatiSolver
            solver = ScipyPositiveRiccatiSolver()

        return solver.solve(equation)


class DefaultPositiveRiccatiSolverLRCF(PositiveRiccatiSolverLRCF):
    r"""Default |PositiveRiccatiSolver|.

    A solver backend is chosen based in the following order:

          1. `slycot` (see :class:`pymor.bindings.slycot.SlycotPositiveRiccatiSolverLRCF`),
          2. `scipy` (see :class:`pymor.bindings.scipy.ScipyPositiveRiccatiSolverLRCF`).

        Currently, only dense solvers are supported.
    """

    def _solve(self, equation):
        if _dense_backend() == 'slycot':
            from pymor.bindings.slycot import SlycotPositiveRiccatiSolverLRCF
            solver = SlycotPositiveRiccatiSolverLRCF()
        else:
            from pymor.bindings.scipy import ScipyPositiveRiccatiSolverLRCF
            solver = ScipyPositiveRiccatiSolverLRCF()

        return solver.solve(equation)


class MatrixEquationSolvers(ImmutableObject):
    """This class configures matrix equation solvers for all supported types of matrix equations.

    Parameters
    ----------
    lyapunov
        A |LyapunovSolver| or `None`, then :class:`DefaultLyapunovSolver` is used.
    lyapunov_lr
        A |LyapunovSolverLRCF| or `None`, then :class:`DefaultLyapunovSolverLRCF` is used.
    riccati
        A |RiccatiSolver| or `None`, then :class:`DefaultRiccatiSolver` is used.
    riccati_lrcf
        A |RiccatiSolverLRCF| or `None`, then :class:`DefaultRiccatiSolverLRCF` is used.
    positive_riccati
        A |PositiveRiccatiSolver| or `None`, then :class:`DefaultPositiveRiccatiSolver`
        is used.
    positive_riccati_lrcf
        A |PositiveRiccatiSolverLRCF| or `None`, then
        :class:`DefaultPositiveRiccatiSolverLRCF` is used.
    """

    def __init__(self, lyapunov=None, lyapunov_lrcf=None, riccati=None, riccati_lrcf=None,
                 positive_riccati=None, positive_riccati_lrcf=None):

        lyapunov = lyapunov or DefaultLyapunovSolver()
        lyapunov_lrcf = lyapunov_lrcf or DefaultLyapunovSolverLRCF()
        riccati = riccati or RiccatiSolver()
        riccati_lrcf = riccati_lrcf or DefaultRiccatiSolverLRCF()
        positive_riccati = positive_riccati or DefaultPositiveRiccatiSolver()
        positive_riccati_lrcf = positive_riccati_lrcf or DefaultPositiveRiccatiSolverLRCF()

        assert isinstance(lyapunov, LyapunovSolver)
        assert isinstance(lyapunov_lrcf, LyapunovSolverLRCF)
        assert isinstance(riccati, RiccatiSolver)
        assert isinstance(riccati_lrcf, RiccatiSolverLRCF)
        assert isinstance(positive_riccati, PositiveRiccatiSolver)
        assert isinstance(positive_riccati_lrcf, PositiveRiccatiSolverLRCF)

        self.__auto_init(locals())

def _dense_backend():
    return 'slycot' if config.HAVE_SLYCOT else 'scipy'

def _warn_dense_fallback(solver, equation, backend):
    if equation.dim >= mat_eqn_sparse_min_size():
        solver.logger.warning(
            f'Using the dense {backend} backend on a {equation.dim} x {equation.dim} problem; '
            'this may be expensive in time and memory.')
