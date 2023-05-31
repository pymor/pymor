# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('PYMEPACK')


import numpy as np
import scipy.linalg as spla
import pymepack

from pymor.algorithms.genericsolvers import _parse_options
from pymor.algorithms.lyapunov import _chol, _solve_lyap_dense_check_args, _solve_lyap_lrcf_check_args
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.logger import getLogger


def lyap_lrcf_solver_options():
    """Return available Lyapunov solvers with default options for the pymepack backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'pymepack_gelyap': {'type': 'pymepack_gelyap'}}

def solve_lyap_lrcf(A, E, B, trans=False, cont_time=True, options=None):
    """Compute an approximate low-rank solution of a Lyapunov equation.

    - :func:`pymor.algorithms.lyapunov.solve_cont_lyap_lrcf`
    - :func:`pymor.algorithms.lyapunov.solve_disc_lyap_lrcf`

    for a general description.

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Lyapunov equation solution, |VectorArray| from `A.source`.
    """
    _solve_lyap_lrcf_check_args(A, E, B, trans)
    options = _parse_options(options, lyap_lrcf_solver_options(), 'pymepack_gelyap', None, False)

    if options['type'] == 'pymepack_gelyap':
        X = solve_lyap_dense(to_matrix(A, format='dense'),
                             to_matrix(E, format='dense') if E else None,
                             B.to_numpy().T if not trans else B.to_numpy(),
                             trans=trans, cont_time=cont_time, options=options)
        Z = _chol(X)
    else:
        raise ValueError(f"Unexpected Lyapunov equation solver ({options['type']}).")

    return A.source.from_numpy(Z.T)

def lyap_dense_solver_options():
    """Return available Lyapunov solvers with default options for the slycot backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'pymepack_gelyap': {'type': 'pymepack_gelyap'}}


def solve_lyap_dense(A, E, B, trans=False, cont_time=True, options=None):
    """Compute the solution of a Lyapunov equation.

    See

    - :func:`pymor.algorithms.lyapunov.solve_cont_lyap_dense`
    - :func:`pymor.algorithms.lyapunov.solve_disc_lyap_dense`

    for a general description.

    This function uses `slycot.sb03md` (if `E is None`) and `slycot.sg03ad` (if `E is not None`),
    which are based on the Bartels-Stewart algorithm.

    Returns
    -------
    X
        Lyapunov equation solution as a |NumPy array|.
    """
    _solve_lyap_dense_check_args(A, E, B, trans)
    options = _parse_options(options, lyap_dense_solver_options(), 'pymepack_gelyap', None, False)

    if options['type'] == 'pymepack_gelyap':
        C = -B.dot(B.T) if not trans else -B.T.dot(B)
        Cf = C if C.flags.f_contiguous else C.copy(order='F')
        Af = A.copy(order='F')
        if E is None:
            pymepack.gelyap(Af, Cf, trans = trans)
        else:
            Ef = E.copy(order='F')
            pymepack.gglyap(Af, Ef, Cf, trans = trans)
    else:
        raise ValueError(f"Unexpected Lyapunov equation solver ({options['type']}).")

    return Cf
