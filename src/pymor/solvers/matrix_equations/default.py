# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


from pymor.core.base import ImmutableObject
from pymor.core.config import config
from pymor.solvers.matrix_equations.interface import (
    LyapunovSolver,
    LyapunovSolverLR,
    PositiveRiccatiSolver,
    PositiveRiccatiSolverLR,
    RiccatiSolver,
    RiccatiSolverLR,
    SylvesterSolver,
)
from pymor.solvers.matrix_equations.utils import mat_eqn_sparse_min_size


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


class DefaultLyapunovSolverLR(LyapunovSolverLR):
    """Default |LyapunovSolverLR|.

    A solver backend is chosen based on availability in the following order:

    - for sparse, continous-time problems (minimum size specified by
        :func:`~pymor.solvers.matrix_equations.utils.mat_eqn_sparse_min_size`)

        1. `lradi` (see :func:`pymor.solvers.matrix_equations.adi.ADILyapunovSolver`),

    - for dense problems (smaller than
        :func:`~pymor.solvers.matrix_equations.utils.mat_eqn_sparse_min_size`)
        or discrete-time problems

        1. `slycot` (see :class:`pymor.bindings.slycot.SlycotLyapunovSolverLR`),
        2. `scipy` (see :class:`pymor.bindings.scipy.ScipyLyapunovSolverLR`).
    """

    def _solve(self, equation):
        backend = _dense_backend() if (not equation.cont_time or equation.dim < mat_eqn_sparse_min_size()) else 'lradi'
        if backend == 'lradi':
            if not equation.cont_time:
                raise ValueError('lradi solves only continuous-time Lyapunov equations.')
            from pymor.solvers.matrix_equations.adi import ADILyapunovSolver
            solver = ADILyapunovSolver()
        else:
            _warn_dense_fallback(self, equation, backend)
            if _dense_backend() == 'slycot':
                from pymor.bindings.slycot import SlycotLyapunovSolverLR
                solver = SlycotLyapunovSolverLR()
            else:
                from pymor.bindings.scipy import ScipyLyapunovSolverLR
                solver = ScipyLyapunovSolverLR()

        return solver.solve(equation)

class DefaultRiccatiSolver(RiccatiSolver):
    r"""Default |RiccatiSolver|.

    A solver backend is chosen based on availability in the following order:

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


class DefaultRiccatiSolverLR(RiccatiSolverLR):
    r"""Default |RiccatiSolverLR|.

    A solver backend is chosen based on availability in the following order:

    - for sparse problems (minimum size specified by
        :func:`~pymor.solvers.matrix_equations.utils.mat_eqn_sparse_min_size`)

        1. `lrradi` (see :class:`pymor.solvers.matrix_equations.radi.RADIRiccatiSolver`),

    - for dense problems (smaller than
        :func:`~pymor.solvers.matrix_equations.utils.mat_eqn_sparse_min_size`)

        1. `slycot` (see :class:`pymor.bindings.slycot.SlycotRiccatiSolverLR`),
        2. `scipy` (see :class:`pymor.bindings.scipy.ScipyRiccatiSolverLR`).
    """

    def _solve(self, equation):
        backend = 'lrradi' if equation.dim >= mat_eqn_sparse_min_size() else _dense_backend()
        if backend == 'lrradi':
            from pymor.solvers.matrix_equations.radi import RADIRiccatiSolver
            solver = RADIRiccatiSolver()
        else:
            _warn_dense_fallback(self, equation, backend)
            if _dense_backend() == 'slycot':
                from pymor.bindings.slycot import SlycotRiccatiSolverLR
                solver = SlycotRiccatiSolverLR()
            else:
                from pymor.bindings.scipy import ScipyRiccatiSolverLR
                solver = ScipyRiccatiSolverLR()

        return solver.solve(equation)


class DefaultPositiveRiccatiSolver(PositiveRiccatiSolver):
    r"""Default |PositiveRiccatiSolver|.

    A solver backend is chosen based on availability in the following order:

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


class DefaultPositiveRiccatiSolverLR(PositiveRiccatiSolverLR):
    r"""Default |PositiveRiccatiSolverLR|.

    A solver backend is chosen based on availability in the following order:

    1. `slycot` (see :class:`pymor.bindings.slycot.SlycotPositiveRiccatiSolverLR`),
    2. `scipy` (see :class:`pymor.bindings.scipy.ScipyPositiveRiccatiSolverLR`).

    Currently, only dense solvers are supported.
    """

    def _solve(self, equation):
        if _dense_backend() == 'slycot':
            from pymor.bindings.slycot import SlycotPositiveRiccatiSolverLR
            solver = SlycotPositiveRiccatiSolverLR()
        else:
            from pymor.bindings.scipy import ScipyPositiveRiccatiSolverLR
            solver = ScipyPositiveRiccatiSolverLR()

        return solver.solve(equation)


class DefaultSylvesterSolver(SylvesterSolver):
    r"""Default |SylvesterSolver|.

    As solver backend the :class:`pymor.solvers.matrix_equations.sylvester.SylvesterSchurSolver`
    is chosen.
    """

    def _solve(self, equation):
        from pymor.solvers.matrix_equations.sylvester import SylvesterSchurSolver
        solver = SylvesterSchurSolver()

        return solver.solve(equation)


class MatrixEquationSolvers(ImmutableObject):
    """This class configures matrix equation solvers for all supported types of matrix equations.

    Parameters
    ----------
    lyapunov
        A |LyapunovSolver| or `None`, then :class:`DefaultLyapunovSolver` is used.
    lyapunov_lr
        A |LyapunovSolverLR| or `None`, then :class:`DefaultLyapunovSolverLR` is used.
    riccati
        A |RiccatiSolver| or `None`, then :class:`DefaultRiccatiSolver` is used.
    riccati_lr
        A |RiccatiSolverLR| or `None`, then :class:`DefaultRiccatiSolverLR` is used.
    positive_riccati
        A |PositiveRiccatiSolver| or `None`, then :class:`DefaultPositiveRiccatiSolver`
        is used.
    positive_riccati_lr
        A |PositiveRiccatiSolverLR| or `None`, then
        :class:`DefaultPositiveRiccatiSolverLR` is used.
    sylvester
        A |SylvesterSolver| or `None`, then :class:`DefaultSylvesterSolver` is used.
    """

    def __init__(self, lyapunov=None, lyapunov_lr=None, riccati=None, riccati_lr=None,
                 positive_riccati=None, positive_riccati_lr=None, sylvester=None):

        lyapunov = lyapunov or DefaultLyapunovSolver()
        lyapunov_lr = lyapunov_lr or DefaultLyapunovSolverLR()
        riccati = riccati or DefaultRiccatiSolver()
        riccati_lr = riccati_lr or DefaultRiccatiSolverLR()
        positive_riccati = positive_riccati or DefaultPositiveRiccatiSolver()
        positive_riccati_lr = positive_riccati_lr or DefaultPositiveRiccatiSolverLR()
        sylvester = sylvester or DefaultSylvesterSolver()

        assert isinstance(lyapunov, LyapunovSolver)
        assert isinstance(lyapunov_lr, LyapunovSolverLR)
        assert isinstance(riccati, RiccatiSolver)
        assert isinstance(riccati_lr, RiccatiSolverLR)
        assert isinstance(positive_riccati, PositiveRiccatiSolver)
        assert isinstance(positive_riccati_lr, PositiveRiccatiSolverLR)
        assert isinstance(sylvester, SylvesterSolver)

        self.__auto_init(locals())

def _dense_backend():
    return 'slycot' if config.HAVE_SLYCOT else 'scipy'

def _warn_dense_fallback(solver, equation, backend):
    if equation.dim >= mat_eqn_sparse_min_size():
        solver.logger.warning(
            f'Using the dense {backend} backend on a {equation.dim} x {equation.dim} problem; '
            'this may be expensive in time and memory.')
