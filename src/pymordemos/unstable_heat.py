#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.sparse as sps
from typer import Argument, run


from pymor.core.config import config
from pymor.core.logger import set_log_levels
from pymor.models.iosys import LTIModel
from pymor.reductors.bt import FDBTReductor
from pymor.reductors.h2 import GapIRKAReductor


def run_mor_method(lti, reductor, reductor_short_name, r, **reduce_kwargs):
    """Run a model order reduction method.

    Parameters
    ----------
    lti
        The full-order |LTIModel|.
    w
        Array of frequencies.
    reductor
        The reductor object.
    reductor_short_name
        A short name for the reductor.
    r
        The order of the reduced-order model.
    reduce_kwargs
        Optional keyword arguments for the reduce method.
    """
    # Reduction
    rom = reductor.reduce(r, **reduce_kwargs)
    err = lti - rom

    # Errors
    if config.HAVE_SLYCOT:
        print(f'{reductor_short_name} relative L_inf-error:  {err.linf_norm() / lti.linf_norm():e}')
    else:
        print('Skipped L_inf-norm calculation due to missing slycot.')


def main(
        l: float = Argument(50, help='Parameter for instability.'),
        r: int = Argument(10, help='Order of the ROMs.'),
):
    r"""1D unstable heat equation demo.

    Discretization of the PDE:

    .. math::

        \begin{align}
            \partial_t T(\xi, t) &= \partial_{\xi \xi} T(\xi, t) + \lambda T(\xi, t),
            & 0 < \xi < 1,\ t > 0, \\
            -\partial_\xi T(0, t) & = -T(0, t) + u(t),
            & t > 0, \\
            \partial_\xi T(1, t) & = -T(1, t),
            & t > 0, \\
            y(t) & = T(1, t),
            & t > 0
        \end{align}

    """
    set_log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING'})

    k = 50
    n = 2 * k + 1

    E = sps.eye(n, format='lil')
    E[0, 0] = E[-1, -1] = 0.5
    E = E.tocsc()

    d0 = n * [-2 * (n - 1)**2 + l]
    d1 = (n - 1) * [(n - 1)**2]
    A = sps.diags([d1, d0, d1], [-1, 0, 1], format='lil')
    A[0, 0] = A[-1, -1] = -n * (n - 1) + l / 2
    A = A.tocsc()

    B = np.zeros((n, 1))
    B[0, 0] = n - 1

    C = np.zeros((1, n))
    C[0, -1] = 1

    lti = LTIModel.from_matrices(A, B, C, E=E)

    ast_spectrum = lti.get_ast_spectrum()
    print(f'Anti-stable system poles:  {" ".join(str(x) for x in ast_spectrum[1])}')

    # Norms of the system
    print(f'FOM L_2-norm:    {lti.l2_norm():e}')
    if config.HAVE_SLYCOT:
        print(f'FOM L_inf-norm:  {lti.hinf_norm():e}')
    else:
        print('Skipped L_inf-norm calculation due to missing slycot.')

    # Model order reduction
    run_mor_method(lti, FDBTReductor(lti), 'FDBT', r, tol=1e-5)
    run_mor_method(lti, GapIRKAReductor(lti), 'GapIRKA', r, tol=1e-5)


if __name__ == '__main__':
    run(main)
