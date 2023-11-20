# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('PYMEPACK')


import pymepack

from pymor.algorithms.genericsolvers import _parse_options
from pymor.algorithms.lyapunov import _chol, _solve_lyap_dense_check_args, _solve_lyap_lrcf_check_args
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.defaults import defaults


@defaults('block_size', 'solver', 'max_it', 'tau')
def pymepack_gelyap_options(block_size = None, solver = None, max_it = None, tau = None):
    """Returns customized options for the pymepack.gelyap, pymepack.gglyap solvers
    and their respective versions with iterative refinement.
    If no values are specified, pymepack uses its default settings.
    Refer to the `pymepack` documentation for default values.

    Parameters
    ----------
    block_size
        See `pymepack.gelyap`, `pymepack.gglyap`.
    solver
        See `pymepack.gelyap`, `pymepack.gglyap`.
    max_it
        See `pymepack.gelyap_refine`, `pymepack.gglyap_refine`.
    tau
        See `pymepack.gelyap_refine`, `pymepack.gglyap_refine`.

    Returns
    -------
    Dictionary of customized solver options to be used instead of the default values for the pymepack backend.
    """
    gelyap_opts = {}

    if block_size is not None: gelyap_opts['block_size'] = block_size
    if solver is not None: gelyap_opts['solver'] = solver
    if max_it is not None: gelyap_refine_opts['max_it'] = max_it
    if tau is not None: gelyap_refine_opts['tau'] = tau

    return gelyap_opts


def lyap_lrcf_solver_options():
    """Return available Lyapunov solvers with preconfigured options for the pymepack backend.

    Returns
    -------
    A dict of available solvers with preconfigured solver options.
    """
    return {
            'pymepack_gelyap': {'type': 'pymepack_gelyap',
                                'opts': pymepack_gelyap_options()},
            'pymepack_gelyap_refine': {'type': 'pymepack_gelyap_refine',
                                       'opts': pymepack_gelyap_options()},
            }

def solve_lyap_lrcf(A, E, B, trans=False, cont_time=True, options=None):
    """Compute an approximate low-rank solution of a Lyapunov equation.

    See

    - :func:`pymor.algorithms.lyapunov.solve_cont_lyap_lrcf`
    - :func:`pymor.algorithms.lyapunov.solve_disc_lyap_lrcf`

    for a general description.

    This function uses `pymepack.gelyap` (if `E is None`) and `pymepack.gglyap` (if `E is not None`),
    which are dense solvers. If options specify a solver with iterative refinement, `pymepack.gelyap_refine`
    and `pymepack.gglyap_refine` are used in the aforementioned cases respectively. We assume A and E can
    be converted to |NumPy arrays| using :func:`~pymor.algorithms.to_matrix.to_matrix` and that
    `B.to_numpy` is implemented.

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
    options = _parse_options(options, lyap_lrcf_solver_options(), 'pymepack_gelyap', None, False)

    if options['type'] in ['pymepack_gelyap', 'pymepack_gelyap_refine']:
        X = solve_lyap_dense(to_matrix(A, format='dense'),
                             to_matrix(E, format='dense') if E else None,
                             B.to_numpy().T if not trans else B.to_numpy(),
                             trans=trans, cont_time=cont_time, options=options)
        Z = _chol(X)
    else:
        raise ValueError(f"Unexpected Lyapunov equation solver ({options['type']}).")

    return A.source.from_numpy(Z.T)

def lyap_dense_solver_options():
    """Return available Lyapunov solvers with preconfigured options for the slycot backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {
            'pymepack_gelyap': {'type': 'pymepack_gelyap',
                                'opts': pymepack_gelyap_options()},
            'pymepack_gelyap_refine': {'type': 'pymepack_gelyap_refine',
                                       'opts': pymepack_gelyap_options()},
            }

def solve_lyap_dense(A, E, B, trans=False, cont_time=True, options=None):
    """Compute the solution of a Lyapunov equation.

    See

    - :func:`pymor.algorithms.lyapunov.solve_cont_lyap_dense`
    - :func:`pymor.algorithms.lyapunov.solve_disc_lyap_dense`

    for a general description.

    In case of the continuous-time Lyapunov equation, this function uses
    `pymepack.gelyap` (if `E is None`) and `pymepack.gglyap` (if `E is not None`),
    which are dense solvers. If options specify a solver with iterative refinement,
    the initial guess is computed with the aforementioned solvers and then refined
    with `pymepack.gelyap_refine` or `pymepack.gglyap_refine` respectively.

    In case of the discrete-time Lyapunov equation, this function uses
    `pymepack.gestein` (if `E is None`) and `pymepack.ggstein` (if `E is not None`),
    which are dense solvers. If options specify a solver with iterative refinement,
    the initial guess is computed with one of the aforementioned solvers and then
    refined with `pymepack.gestein_refine` or `pymepack.ggstein_refine` respectively.


    Parameters
    ----------
    A
        The matrix A as a 2D |NumPy array|.
    E
        The matrix E as a 2D |NumPy array| or `None`.
    B
        The matrix B as a 2D |NumPy array|.
    trans
        Whether the first matrix in the Lyapunov equation is transposed.
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
    options = _parse_options(options, lyap_dense_solver_options(), 'pymepack_gelyap', None, False)

    C = -B.dot(B.T) if not trans else -B.T.dot(B)
    Cf = C if C.flags.f_contiguous else C.copy(order='F')
    Af = A.copy(order='F')
    X = None

    opts = options['opts']
    refinement_opts = {}
    if 'max_it' in opts : refinement_opts['max_it'] = opts.pop('max_it')
    if 'tau' in opts : refinement_opts['tau'] = opts.pop('tau')
    if 'block_size' in opts: refinement_opts['block_size'] = opts['block_size']
    if 'solver' in opts : refinement_opts['solver'] = opts['solver']

    if options['type'] == 'pymepack_gelyap':
        if E is None:
            if cont_time:
                pymepack.gelyap(Af, Cf, trans = trans, inplace = True, **opts)
            else:
                pymepack.gestein(Af, Cf, trans = trans, inplace = True, **opts)
        else:
            Ef = E.copy(order='F')
            if cont_time:
                pymepack.gglyap(Af, Ef, Cf, trans = trans, inplace = True, **opts)
            else:
                pymepack.ggstein(Af, Ef, Cf, trans = trans, inplace = True, **opts)
        X = Cf
    elif options['type'] == 'pymepack_gelyap_refine':
        if E is None:
            if cont_time:
                X,AS,Q = pymepack.gelyap(Af, Cf, trans = trans, **opts, inplace = False)
                X, *_ = pymepack.gelyap_refine(Af, Cf, AS, Q, X, trans = trans, **refinement_opts)
            else:
                X,AS,Q = pymepack.gestein(Af, Cf, trans = trans, **opts, inplace = False)
                X, *_ = pymepack.gestein_refine(Af, Cf, AS, Q, X, trans = trans, **refinement_opts)
        else:
            Ef = E if E.flags.f_contiguous else E.copy(order='F')
            if cont_time:
                X,AS,ES,Q,Z = pymepack.gglyap(Af, Ef, Cf, trans = trans, **opts, inplace = False)
                opts
                X, *_ = pymepack.gglyap_refine(Af, Ef, Cf, AS, ES, Q, Z, X, trans = trans, **refinement_opts)
            else:
                X,AS,ES,Q,Z = pymepack.ggstein(Af, Ef, Cf, trans = trans, **opts, inplace = False)
                X, *_ = pymepack.ggstein_refine(Af, Ef, Cf, AS, ES, Q, Z, X, trans = trans, **refinement_opts)
    else:
        raise ValueError(f"Unexpected Lyapunov equation solver ({options['type']}).")

    return X
