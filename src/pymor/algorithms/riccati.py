# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
from pymor.core.defaults import defaults


_DEFAULT_SOLVER = 'slycot' if config.HAVE_SLYCOT else \
                  'pymess' if config.HAVE_PYMESS else \
                  'scipy'


@defaults('me_solver')
def solve_ricc(A, E=None, B=None, Q=None, C=None, R=None, G=None,
               trans=False, me_solver=_DEFAULT_SOLVER, tol=None):
    """Find a factor of the solution of a Riccati equation

    Returns factor :math:`Z` such that :math:`Z Z^T` is approximately the
    solution :math:`X` of a Riccati equation

    .. math::
        A^T X E + E^T X A - E^T X B R^{-1} B^T X E + Q = 0.

    If E in `None`, it is taken to be the identity matrix.
    Q can instead be given as C^T * C. In this case, Q needs to be `None`, and
    C not `None`.
    B * R^{-1} B^T can instead be given by G. In this case, B and R need to be
    `None`, and G not `None`.
    If R and G are `None`, then R is taken to be the identity matrix.
    If trans is `True`, then the dual Riccati equation is solved

    .. math::
        A X E^T + E X A^T - E X C^T R^{-1} C X E^T + Q = 0,

    where Q can be replaced by B * B^T and C^T * R^{-1} * C by G.

    Parameters
    ----------
    A
        The |Operator| A.
    B
        The |Operator| B or `None`.
    E
        The |Operator| E or `None`.
    Q
        The |Operator| Q or `None`.
    C
        The |Operator| C or `None`.
    R
        The |Operator| R or `None`.
    D
        The |Operator| D or `None`.
    G
        The |Operator| G or `None`.
    L
        The |Operator| L or `None`.
    trans
        If the dual equation needs to be solved.
    me_solver
        Method to use ('scipy', 'slycot', 'pymess', 'pymess_care', 'pymess_lrnm').
    tol
        Tolerance parameter.

    Returns
    -------
    Z
        Low-rank factor of the Riccati equation solution,
        |VectorArray| from `A.source`.
    """
    assert me_solver in {'scipy', 'slycot', 'pymess', 'pymess_care', 'pymess_lrnm'}

    if me_solver == 'scipy':
        from pymor.bindings.scipy import solve_ricc_impl
    elif me_solver == 'slycot':
        from pymor.bindings.slycot import solve_ricc as solve_ricc_impl
    elif me_solver.startswith('pymess'):
        from pymor.bindings.pymess import solve_ricc as solve_ricc_impl
    else:
        assert False

    return solve_ricc_impl(A, E=E, B=B, Q=Q, C=C, R=R, G=G, trans=trans, me_solver=me_solver, tol=tol)
