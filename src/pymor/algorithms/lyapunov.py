# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)


from pymor.core.config import config
from pymor.core.defaults import defaults


_DEFAULT_SOLVER = 'slycot' if config.HAVE_SLYCOT else \
                  'pymess' if config.HAVE_PYMESS else \
                  'scipy'


@defaults('me_solver')
def solve_lyap(A, E, B, trans=False, me_solver=_DEFAULT_SOLVER, tol=None):
    """Find a factor of the solution of a Lyapunov equation

    Returns factor :math:`Z` such that :math:`Z Z^T` is approximately
    the solution :math:`X` of a Lyapunov equation (if E is `None`)

    .. math::
        A X + X A^T + B B^T = 0

    or generalized Lyapunov equation

    .. math::
        A X E^T + E X A^T + B B^T = 0.

    If trans is `True`, then solve (if E is `None`)

    .. math::
        A^T X + X A + B^T B = 0

    or

    .. math::
        A^T X E + E^T X A + B^T B = 0.

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E or `None`.
    B
        The |Operator| B.
    trans
        If the dual equation needs to be solved.
    me_solver
        Solver to use ('scipy', 'slycot', 'pymess', 'pymess_lyap', 'pymess_lradi').
    tol
        Tolerance parameter.

    Returns
    -------
    Z
        Low-rank factor of the Lyapunov equation solution, |VectorArray| from `A.source`.
    """
    from pymor.bindings.scipy import _solve_lyap_check_args
    _solve_lyap_check_args(A, E, B, trans)
    assert me_solver in ('scipy', 'slycot', 'pymess', 'pymess_lyap', 'pymess_lradi')

    if me_solver == 'scipy':
        from pymor.bindings.scipy import solve_lyap as solve_lyap_impl
    elif me_solver == 'slycot':
        from pymor.bindings.slycot import solve_lyap as solve_lyap_impl
    elif me_solver.startswith('pymess'):
        from pymor.bindings.pymess import solve_lyap as solve_lyap_impl
    else:
        assert False

    return solve_lyap_impl(A, E, B, trans=trans, me_solver=me_solver, tol=tol)
