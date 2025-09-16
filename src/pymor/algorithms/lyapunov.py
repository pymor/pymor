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
                'dense': 'slycot' if config.HAVE_SLYCOT else 'scipy',
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
def solve_cont_lyap_lrcf(A, E, B, trans=False, options=None,
                         default_sparse_solver_backend=_DEFAULT_LYAP_SOLVER_BACKEND['cont']['sparse'],
                         default_dense_solver_backend=_DEFAULT_LYAP_SOLVER_BACKEND['cont']['dense']):
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        if A.source.dim >= mat_eqn_sparse_min_size():
            backend = default_sparse_solver_backend
        else:
            backend = default_dense_solver_backend
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'lradi':
        from pymor.algorithms.lradi import solve_lyap_lrcf as solve_lyap_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    return solve_lyap_impl(A, E, B, trans=trans, cont_time=True, options=options)


@defaults('default_dense_solver_backend')
def solve_disc_lyap_lrcf(A, E, B, trans=False, options=None,
                         default_dense_solver_backend=_DEFAULT_LYAP_SOLVER_BACKEND['disc']['dense']):
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        backend = default_dense_solver_backend
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_lyap_lrcf as solve_lyap_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    return solve_lyap_impl(A, E, B, trans=trans, cont_time=False, options=options)


@defaults('default_solver_backend')
def solve_cont_lyap_dense(A, E, B, trans=False, options=None,
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

    We assume A and E are real |NumPy arrays|, E is invertible, and that no two eigenvalues of
    (A, E) sum to zero (i.e., there exists a unique solution X).

    If the solver is not specified using the options argument, a solver backend is chosen based on
    availability in the following order:

    1. `slycot` (see :func:`pymor.bindings.slycot.solve_lyap_dense`)
    2. `scipy` (see :func:`pymor.bindings.scipy.solve_lyap_dense`)

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
    options
        The solver options to use.
        See:

        - :func:`pymor.bindings.scipy.lyap_dense_solver_options`,
        - :func:`pymor.bindings.slycot.lyap_dense_solver_options`,

    default_solver_backend
        Default solver backend to use (slycot, scipy).

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
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_lyap_dense as solve_lyap_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_lyap_dense as solve_lyap_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    return solve_lyap_impl(A, E, B, trans=trans, cont_time=True, options=options)


@defaults('default_solver_backend')
def solve_disc_lyap_dense(A, E, B, trans=False, options=None,
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

    We assume A and E are real |NumPy arrays|, E is invertible, and that all pairwise products of
    two eigenvalues of (A, E) are not equal to one (i.e., there exists a unique solution X).

    If the solver is not specified using the options argument, a solver backend is chosen based on
    availability in the following order:

    1. `slycot` (see :func:`pymor.bindings.slycot.solve_lyap_dense`)
    2. `scipy` (see :func:`pymor.bindings.scipy.solve_lyap_dense`)

    Parameters
    ----------
    A
        The matrix A as a 2D |NumPy array|.
    E
        The matrix E as a 2D |NumPy array| or `None`.
    B
        The matrix B as a 2D |NumPy array|.
    trans
        Whether the first operator in the Lyapunov equation is
        transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.bindings.scipy.lyap_dense_solver_options`,
        - :func:`pymor.bindings.slycot.lyap_dense_solver_options`.

    default_solver_backend
        Default solver backend to use (slycot, scipy).

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
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_lyap_dense as solve_lyap_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_lyap_dense as solve_lyap_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    return solve_lyap_impl(A, E, B, trans=trans, cont_time=False, options=options)


