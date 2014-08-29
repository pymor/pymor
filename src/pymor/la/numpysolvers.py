# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import bicgstab, spsolve, spilu, LinearOperator

from pymor.core.defaults import defaults, defaults_sid
from pymor.core.exceptions import InversionError

try:
    import pyamg
    HAVE_PYAMG = True
except ImportError:
    HAVE_PYAMG = False


_dense_options = None
_dense_options_sid = None
_sparse_options = None
_sparse_options_sid = None


@defaults('default_solver', 'least_squares_rcond')
def dense_options(default_solver='solve',
                  least_squares_rcond=-1.):
    opts = (('solve',         {'type': 'solve'}),
            ('least_squares', {'type': 'least_squares',
                               'rcond': -1.}))
    opts = OrderedDict(opts)
    def_opt = opts.pop(default_solver)
    ordered_opts = OrderedDict(((default_solver, def_opt),))
    ordered_opts.update(opts)
    return ordered_opts


@defaults('default_solver', 'bicgstab_tol', 'bicgstab_maxiter', 'spilu_drop_tol',
          'spilu_fill_factor', 'spilu_drop_rule', 'spilu_permc_spec', 'spsolve_permc_spec',
          'pyamg_tol', 'pyamg_maxiter', 'pyamg_verb', 'pyamg_rs_strength', 'pyamg_rs_CF',
          'pyamg_rs_postsmoother', 'pyamg_rs_max_levels', 'pyamg_rs_max_coarse', 'pyamg_rs_coarse_solver',
          'pyamg_rs_cycle', 'pyamg_rs_accel', 'pyamg_rs_tol', 'pyamg_rs_maxiter',
          'pyamg_sa_symmetry', 'pyamg_sa_strength', 'pyamg_sa_aggregate', 'pyamg_sa_smooth',
          'pyamg_sa_presmoother', 'pyamg_sa_postsmoother', 'pyamg_sa_improve_candidates', 'pyamg_sa_max_levels',
          'pyamg_sa_max_coarse', 'pyamg_sa_diagonal_dominance', 'pyamg_sa_coarse_solver', 'pyamg_sa_cycle',
          'pyamg_sa_accel', 'pyamg_sa_tol', 'pyamg_sa_maxiter')
def sparse_options(default_solver='spsolve',
                   bicgstab_tol=1e-15,
                   bicgstab_maxiter=None,
                   spilu_drop_tol=1e-4,
                   spilu_fill_factor=10,
                   spilu_drop_rule='basic,area',
                   spilu_permc_spec='COLAMD',
                   spsolve_permc_spec='COLAMD',
                   pyamg_tol=1e-5,
                   pyamg_maxiter=400,
                   pyamg_verb=False,
                   pyamg_rs_strength=('classical', {'theta': 0.25}),
                   pyamg_rs_CF='RS',
                   pyamg_rs_presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                   pyamg_rs_postsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                   pyamg_rs_max_levels=10,
                   pyamg_rs_max_coarse=500,
                   pyamg_rs_coarse_solver='pinv2',
                   pyamg_rs_cycle='V',
                   pyamg_rs_accel=None,
                   pyamg_rs_tol=1e-5,
                   pyamg_rs_maxiter=100,
                   pyamg_sa_symmetry='hermitian',
                   pyamg_sa_strength='symmetric',
                   pyamg_sa_aggregate='standard',
                   pyamg_sa_smooth=('jacobi', {'omega': 4.0/3.0}),
                   pyamg_sa_presmoother=('block_gauss_seidel', {'sweep': 'symmetric'}),
                   pyamg_sa_postsmoother=('block_gauss_seidel', {'sweep': 'symmetric'}),
                   pyamg_sa_improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None],
                   pyamg_sa_max_levels=10,
                   pyamg_sa_max_coarse=500,
                   pyamg_sa_diagonal_dominance=False,
                   pyamg_sa_coarse_solver='pinv2',
                   pyamg_sa_cycle='V',
                   pyamg_sa_accel=None,
                   pyamg_sa_tol=1e-5,
                   pyamg_sa_maxiter=100):
    """Returns |invert_options| (with default values) for sparse |NumPy| matricies.

    Parameters
    ----------
    default_solver
        Default sparse solver to use (spsolve, bicgstab, bicgstab-spilu, pyamg,
        pyamg_rs, pyamg_sa).
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
    pyamg_tol
        Tolerance for `PyAMG <http://pyamg.github.io/>`_ blackbox solver.
    pyamg_maxiter
        Maximum iterations for `PyAMG <http://pyamg.github.io/>`_ blackbox solver.
    pyamg_verb
        Verbosity flag for `PyAMG <http://pyamg.github.io/>`_ blackbox solver.
    pyamg_rs_strength
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_CF
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_presmoother
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_postsmoother
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_max_levels
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_max_coarse
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_coarse_solver
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_cycle
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_accel
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_tol
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_maxiter
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_sa_symmetry
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_strength
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_aggregate
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_smooth
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_presmoother
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_postsmoother
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_improve_candidates
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_max_levels
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_max_coarse
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_diagonal_dominance
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_coarse_solver
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_cycle
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_accel
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_tol
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_maxiter
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.

    Returns
    -------
    A tuple of all possible |invert_options|.
    """

    opts = (('bicgstab-spilu', {'type': 'bicgstab-spilu',
                                'tol': bicgstab_tol,
                                'maxiter': bicgstab_maxiter,
                                'spilu_drop_tol': spilu_drop_tol,
                                'spilu_fill_factor': spilu_fill_factor,
                                'spilu_drop_rule': spilu_drop_rule,
                                'spilu_permc_spec': spilu_permc_spec}),
            ('bicgstab',       {'type': 'bicgstab',
                                'tol': bicgstab_tol,
                                'maxiter': bicgstab_maxiter}),
            ('spsolve',        {'type': 'spsolve',
                                'permc_spec': spsolve_permc_spec}))
    if HAVE_PYAMG:
        opts += (('pyamg',    {'type': 'pyamg',
                               'tol': pyamg_tol,
                               'maxiter': pyamg_maxiter}),
                 ('pyamg-rs', {'type': 'pyamg-rs',
                               'strength': pyamg_rs_strength,
                               'CF': pyamg_rs_CF,
                               'presmoother': pyamg_rs_presmoother,
                               'postsmoother': pyamg_rs_postsmoother,
                               'max_levels': pyamg_rs_max_levels,
                               'max_coarse': pyamg_rs_max_coarse,
                               'coarse_solver': pyamg_rs_coarse_solver,
                               'cycle': pyamg_rs_cycle,
                               'accel': pyamg_rs_accel,
                               'tol': pyamg_rs_tol,
                               'maxiter': pyamg_rs_maxiter}),
                 ('pyamg-sa', {'type': 'pyamg-sa',
                               'symmetry': pyamg_sa_symmetry,
                               'strength': pyamg_sa_strength,
                               'aggregate': pyamg_sa_aggregate,
                               'smooth': pyamg_sa_smooth,
                               'presmoother': pyamg_sa_presmoother,
                               'postsmoother': pyamg_sa_postsmoother,
                               'improve_candidates': pyamg_sa_improve_candidates,
                               'max_levels': pyamg_sa_max_levels,
                               'max_coarse': pyamg_sa_max_coarse,
                               'diagonal_dominance': pyamg_sa_diagonal_dominance,
                               'coarse_solver': pyamg_sa_coarse_solver,
                               'cycle': pyamg_sa_cycle,
                               'accel': pyamg_sa_accel,
                               'tol': pyamg_sa_tol,
                               'maxiter': pyamg_sa_maxiter}))
    opts = OrderedDict(opts)
    def_opt = opts.pop(default_solver)
    ordered_opts = OrderedDict(((default_solver, def_opt),))
    ordered_opts.update(opts)
    return ordered_opts


def invert_options(matrix=None, sparse=None):
    """Returns |invert_options| (with default values) for a given |NumPy| matrix.

    See :func:`sparse_options` for documentation of all possible options for
    sparse matrices.

    Parameters
    ----------
    matrix
        The matrix for which to return the options.
    sparse
        Instead of providing a matrix via the `matrix` argument,
        `sparse` can be set to `True` or `False` to requset the
        invert options for sparse or dense matrices.

    Returns
    -------
    A tuple of all possible |invert_options|.
    """
    global _dense_options, _dense_options_sid, _sparse_options, _sparse_options_sid
    assert (matrix is None) != (sparse is None)
    sparse = sparse if sparse is not None else issparse(matrix)
    if sparse:
        if not _sparse_options or _sparse_options_sid != defaults_sid():
            _sparse_options = sparse_options()
            _sparse_options_sid = defaults_sid()
            return _sparse_options
        else:
            return _sparse_options
    else:
        if not _dense_options or _dense_options_sid != defaults_sid():
            _dense_options = dense_options()
            _dense_options_sid = defaults_sid()
            return _dense_options
        else:
            return _dense_options


def apply_inverse(matrix, U, options=None):
    """Solve linear equation system.

    Applies the inverse of `matrix` to the row vectors in `U`.

    See :func:`sparse_options` for documentation of all possible options for
    sparse matrices.

    Parameters
    ----------
    matrix
        The |NumPy| matrix to invert.
    U
        2-dimensional |NumPy array| containing as row vectors
        the right-hand sides of the linear equation systems to
        solve.
    options
        |invert_options| to use. (See :func:`invert_options`.)

    Returns
    -------
    |NumPy array| of the solution vectors.
    """

    default_options = invert_options(matrix)

    if options is None:
        options = default_options.values()[0]
    elif isinstance(options, str):
        options = default_options[options]
    else:
        assert 'type' in options and options['type'] in default_options \
            and options.viewkeys() <= default_options[options['type']].viewkeys()
        user_options = options
        options = default_options[user_options['type']]
        options.update(user_options)

    R = np.empty((len(U), matrix.shape[1]))

    if options['type'] == 'solve':
        for i, UU in enumerate(U):
            try:
                R[i] = np.linalg.solve(matrix, UU)
            except np.linalg.LinAlgError as e:
                raise InversionError('{}: {}'.format(str(type(e)), str(e)))
    elif options['type'] == 'least_squares':
        for i, UU in enumerate(U):
            try:
                R[i], _, _, _ = np.linalg.lstsq(matrix, UU, rcond=options['rcond'])
            except np.linalg.LinAlgError as e:
                raise InversionError('{}: {}'.format(str(type(e)), str(e)))
    elif options['type'] == 'bicgstab':
        for i, UU in enumerate(U):
            R[i], info = bicgstab(matrix, UU, tol=options['tol'], maxiter=options['maxiter'])
            if info != 0:
                if info > 0:
                    raise InversionError('bicgstab failed to converge after {} iterations'.format(info))
                else:
                    raise InversionError('bicgstab failed with error code {} (illegal input or breakdown)'.
                                         format(info))
    elif options['type'] == 'bicgstab-spilu':
        ilu = spilu(matrix, drop_tol=options['spilu_drop_tol'], fill_factor=options['spilu_fill_factor'],
                    drop_rule=options['spilu_drop_rule'], permc_spec=options['spilu_permc_spec'])
        precond = LinearOperator(matrix.shape, ilu.solve)
        for i, UU in enumerate(U):
            R[i], info = bicgstab(matrix, UU, tol=options['tol'], maxiter=options['maxiter'], M=precond)
            if info != 0:
                if info > 0:
                    raise InversionError('bicgstab failed to converge after {} iterations'.format(info))
                else:
                    raise InversionError('bicgstab failed with error code {} (illegal input or breakdown)'.
                                         format(info))
    elif options['type'] == 'spsolve':
        for i, UU in enumerate(U):
            R[i] = spsolve(matrix, UU, permc_spec=options['permc_spec'])
    elif options['type'] == 'pyamg':
        if len(U) > 0:
            U_iter = iter(enumerate(U))
            R[0], ml = pyamg.solve(matrix, next(U_iter)[1],
                                   tol=options['tol'],
                                   maxiter=options['maxiter'],
                                   return_solver=True)
            for i, UU in U_iter:
                R[i] = pyamg.solve(matrix, UU,
                                   tol=options['tol'],
                                   maxiter=options['maxiter'],
                                   existing_solver=ml)
    elif options['type'] == 'pyamg-rs':
        ml = pyamg.ruge_stuben_solver(matrix,
                                      strength=options['strength'],
                                      CF=options['CF'],
                                      presmoother=options['presmoother'],
                                      postsmoother=options['postsmoother'],
                                      max_levels=options['max_levels'],
                                      max_coarse=options['max_coarse'],
                                      coarse_solver=options['coarse_solver'])
        for i, UU in enumerate(U):
            R[i] = ml.solve(UU,
                            tol=options['tol'],
                            maxiter=options['maxiter'],
                            cycle=options['cycle'],
                            accel=options['accel'])
    elif options['type'] == 'pyamg-sa':
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
        for i, UU in enumerate(U):
            R[i] = ml.solve(UU,
                            tol=options['tol'],
                            maxiter=options['maxiter'],
                            cycle=options['cycle'],
                            accel=options['accel'])
    else:
        raise ValueError('Unknown solver type')
    return R
