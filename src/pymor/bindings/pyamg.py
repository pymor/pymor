# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
from pymor.tools.deprecated import Deprecated

if config.HAVE_PYAMG:

    import numpy as np
    import pyamg

    from pymor.algorithms.genericsolvers import _parse_options
    from pymor.core.defaults import defaults
    from pymor.core.exceptions import InversionError
    from pymor.operators.numpy import NumpyMatrixOperator

    @defaults('tol', 'maxiter', 'verb', 'rs_strength', 'rs_CF',
              'rs_postsmoother', 'rs_max_levels', 'rs_max_coarse', 'rs_coarse_solver',
              'rs_cycle', 'rs_accel', 'rs_tol', 'rs_maxiter',
              'sa_symmetry', 'sa_strength', 'sa_aggregate', 'sa_smooth',
              'sa_presmoother', 'sa_postsmoother', 'sa_improve_candidates', 'sa_max_levels',
              'sa_max_coarse', 'sa_diagonal_dominance', 'sa_coarse_solver', 'sa_cycle',
              'sa_accel', 'sa_tol', 'sa_maxiter')
    def solver_options(tol=1e-5,
                       maxiter=400,
                       verb=False,
                       rs_strength=('classical', {'theta': 0.25}),
                       rs_CF='RS',
                       rs_presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                       rs_postsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                       rs_max_levels=10,
                       rs_max_coarse=500,
                       rs_coarse_solver='pinv2',
                       rs_cycle='V',
                       rs_accel=None,
                       rs_tol=1e-5,
                       rs_maxiter=100,
                       sa_symmetry='hermitian',
                       sa_strength='symmetric',
                       sa_aggregate='standard',
                       sa_smooth=('jacobi', {'omega': 4.0/3.0}),
                       sa_presmoother=('block_gauss_seidel', {'sweep': 'symmetric'}),
                       sa_postsmoother=('block_gauss_seidel', {'sweep': 'symmetric'}),
                       sa_improve_candidates=(('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None),
                       sa_max_levels=10,
                       sa_max_coarse=500,
                       sa_diagonal_dominance=False,
                       sa_coarse_solver='pinv2',
                       sa_cycle='V',
                       sa_accel=None,
                       sa_tol=1e-5,
                       sa_maxiter=100):
        """Returns available solvers with default |solver_options| for the PyAMG backend.

        Parameters
        ----------
        tol
            Tolerance for `PyAMG <http://pyamg.github.io/>`_ blackbox solver.
        maxiter
            Maximum iterations for `PyAMG <http://pyamg.github.io/>`_ blackbox solver.
        verb
            Verbosity flag for `PyAMG <http://pyamg.github.io/>`_ blackbox solver.
        rs_strength
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
        rs_CF
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
        rs_presmoother
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
        rs_postsmoother
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
        rs_max_levels
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
        rs_max_coarse
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
        rs_coarse_solver
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
        rs_cycle
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
        rs_accel
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
        rs_tol
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
        rs_maxiter
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
        sa_symmetry
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
        sa_strength
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
        sa_aggregate
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
        sa_smooth
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
        sa_presmoother
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
        sa_postsmoother
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
        sa_improve_candidates
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
        sa_max_levels
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
        sa_max_coarse
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
        sa_diagonal_dominance
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
        sa_coarse_solver
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
        sa_cycle
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
        sa_accel
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
        sa_tol
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
        sa_maxiter
            Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.

        Returns
        -------
        A dict of available solvers with default |solver_options|.
        """
        return {'pyamg_solve': {'type': 'pyamg_solve',
                                'tol': tol,
                                'maxiter': maxiter},
                'pyamg_rs':    {'type': 'pyamg_rs',
                                'strength': rs_strength,
                                'CF': rs_CF,
                                'presmoother': rs_presmoother,
                                'postsmoother': rs_postsmoother,
                                'max_levels': rs_max_levels,
                                'max_coarse': rs_max_coarse,
                                'coarse_solver': rs_coarse_solver,
                                'cycle': rs_cycle,
                                'accel': rs_accel,
                                'tol': rs_tol,
                                'maxiter': rs_maxiter},
                'pyamg_sa':    {'type': 'pyamg_sa',
                                'symmetry': sa_symmetry,
                                'strength': sa_strength,
                                'aggregate': sa_aggregate,
                                'smooth': sa_smooth,
                                'presmoother': sa_presmoother,
                                'postsmoother': sa_postsmoother,
                                'improve_candidates': sa_improve_candidates,
                                'max_levels': sa_max_levels,
                                'max_coarse': sa_max_coarse,
                                'diagonal_dominance': sa_diagonal_dominance,
                                'coarse_solver': sa_coarse_solver,
                                'cycle': sa_cycle,
                                'accel': sa_accel,
                                'tol': sa_tol,
                                'maxiter': sa_maxiter}}

    @Deprecated('pyamg bindings will be removed after the 2021.1 release')
    @defaults('check_finite', 'default_solver')
    def apply_inverse(op, V, initial_guess=None, options=None, least_squares=False,
                      check_finite=True, default_solver='pyamg_solve'):
        """Solve linear equation system.

        Applies the inverse of `op` to the vectors in `V` using PyAMG.

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
            Must be `False`.
        check_finite
            Test if solution only contains finite values.
        default_solver
            Default solver to use (pyamg_solve, pyamg_rs, pyamg_sa).

        Returns
        -------
        |VectorArray| of the solution vectors.
        """
        assert V in op.range
        assert initial_guess is None or initial_guess in op.source and len(initial_guess) == len(V)

        if least_squares:
            raise NotImplementedError

        if isinstance(op, NumpyMatrixOperator):
            matrix = op.matrix
        else:
            from pymor.algorithms.to_matrix import to_matrix
            matrix = to_matrix(op)

        options = _parse_options(options, solver_options(), default_solver, None, least_squares)

        V = V.to_numpy()
        promoted_type = np.promote_types(matrix.dtype, V.dtype)
        R = np.empty((len(V), matrix.shape[1]), dtype=promoted_type)

        if options['type'] == 'pyamg_solve':
            if len(V) > 0:
                V_iter = iter(enumerate(V))
                R[0], ml = pyamg.solve(matrix, next(V_iter)[1],
                                       tol=options['tol'],
                                       maxiter=options['maxiter'],
                                       return_solver=True)
                for i, VV in V_iter:
                    R[i] = pyamg.solve(matrix, VV,
                                       tol=options['tol'],
                                       maxiter=options['maxiter'],
                                       existing_solver=ml)
        elif options['type'] == 'pyamg_rs':
            ml = pyamg.ruge_stuben_solver(matrix,
                                          strength=options['strength'],
                                          CF=options['CF'],
                                          presmoother=options['presmoother'],
                                          postsmoother=options['postsmoother'],
                                          max_levels=options['max_levels'],
                                          max_coarse=options['max_coarse'],
                                          coarse_solver=options['coarse_solver'])
            for i, VV in enumerate(V):
                R[i] = ml.solve(VV,
                                tol=options['tol'],
                                maxiter=options['maxiter'],
                                cycle=options['cycle'],
                                accel=options['accel'])
        elif options['type'] == 'pyamg_sa':
            ml = pyamg.smoothed_aggregation_solver(matrix,
                                                   symmetry=options['symmetry'],
                                                   strength=options['strength'],
                                                   aggregate=options['aggregate'],
                                                   smooth=options['smooth'],
                                                   presmoother=options['presmoother'],
                                                   postsmoother=options['postsmoother'],
                                                   improve_candidates=options['improve_candidates'],
                                                   max_levels=options['max_levels'],
                                                   max_coarse=options['max_coarse'],
                                                   diagonal_dominance=options['diagonal_dominance'])
            for i, VV in enumerate(V):
                R[i] = ml.solve(VV,
                                tol=options['tol'],
                                maxiter=options['maxiter'],
                                cycle=options['cycle'],
                                accel=options['accel'])
        else:
            raise ValueError('Unknown solver type')

        if check_finite:
            if not np.isfinite(np.sum(R)):
                raise InversionError('Result contains non-finite values')

        return op.source.from_numpy(R)
