#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from typer import Argument, run


from pymor.core.logger import set_log_levels
from pymor.models.iosys import LTIModel
from pymor.reductors.bt import FDBTReductor
from pymor.reductors.h2 import GapIRKAReductor
from pymordemos.heat import fom_properties, run_mor_method


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
    set_log_levels({
        'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING',
        'pymor.algorithms.lradi.solve_lyap_lrcf': 'WARNING',
        'pymor.reductors.basic.LTIPGReductor': 'WARNING',
    })
    plt.rcParams['axes.grid'] = True

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

    # LTI system
    lti = LTIModel.from_matrices(A, B, C, E=E)

    # Figure
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    subfigs = fig.subfigures(1, 2)
    fig.suptitle('Full-order model')

    # System properties
    w = np.logspace(-1, 3, 100)
    w = (1e-1, 1e3)
    fom_properties(lti, w, stable=False, fig_bode=subfigs[0], fig_poles=subfigs[1])
    plt.show()

    # Model order reduction
    run_mor_method(lti, w, FDBTReductor(lti), 'FDBT', r, stable=False, tol=1e-5)
    run_mor_method(lti, w, GapIRKAReductor(lti), 'GapIRKA', r, stable=False, tol=1e-5)


if __name__ == '__main__':
    run(main)
