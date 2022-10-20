#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from typer import Argument, run

from pymor.core.logger import set_log_levels
from pymor.models.iosys import SecondOrderModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.bt import BTReductor
from pymor.reductors.h2 import IRKAReductor
from pymor.reductors.mt import MTReductor
from pymor.reductors.sobt import (SOBTpReductor, SOBTvReductor, SOBTpvReductor, SOBTvpReductor,
                                  SOBTfvReductor, SOBTReductor)
from pymor.reductors.sor_irka import SORIRKAReductor
from pymordemos.parametric_heat import fom_properties_param, run_mor_method_param


def main(
        n: int = Argument(101, help='Order of the full second-order model (odd number).'),
        r: int = Argument(5, help='Order of the ROMs.'),
):
    """Parametric string example."""
    set_log_levels({
        'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING',
        'pymor.algorithms.lradi.solve_lyap_lrcf': 'WARNING',
        'pymor.reductors.basic.LTIPGReductor': 'WARNING',
    })
    plt.rcParams['axes.grid'] = True

    # Assemble M, D, K, B, C_p
    assert n % 2 == 1, 'The order has to be an odd integer.'

    n2 = (n + 1) // 2

    k = 0.01  # stiffness

    M = sps.eye(n, format='csc')
    E = sps.eye(n, format='csc')
    K = sps.diags([n * [2 * k * n ** 2],
                   (n - 1) * [-k * n ** 2],
                   (n - 1) * [-k * n ** 2]],
                  [0, -1, 1],
                  format='csc')
    B = np.zeros((n, 1))
    B[n2 - 1, 0] = n
    Cp = np.zeros((1, n))
    Cp[0, n2 - 1] = 1

    # Second-order system
    Mop = NumpyMatrixOperator(M)
    Eop = NumpyMatrixOperator(E) * ProjectionParameterFunctional('damping')
    Kop = NumpyMatrixOperator(K)
    Bop = NumpyMatrixOperator(B)
    Cpop = NumpyMatrixOperator(Cp)

    so_sys = SecondOrderModel(Mop, Eop, Kop, Bop, Cpop)

    mus = [1, 5, 10]
    w = (1e-3, 1e2)
    fom_properties_param(so_sys, w, mus)

    # Model order reduction
    run_mor_method_param(so_sys, r, w, mus, SOBTpReductor, 'SOBTp')
    run_mor_method_param(so_sys, r, w, mus, SOBTvReductor, 'SOBTv')
    run_mor_method_param(so_sys, r, w, mus, SOBTpvReductor, 'SOBTpv')
    run_mor_method_param(so_sys, r, w, mus, SOBTvpReductor, 'SOBTvp')
    run_mor_method_param(so_sys, r, w, mus, SOBTfvReductor, 'SOBTfv')
    run_mor_method_param(so_sys, r, w, mus, SOBTReductor, 'SOBT')
    run_mor_method_param(so_sys, r, w, mus, SORIRKAReductor, 'SOR-IRKA')
    run_mor_method_param(so_sys.to_lti(), r, w, mus, BTReductor, 'BT')
    run_mor_method_param(so_sys.to_lti(), r, w, mus, IRKAReductor, 'IRKA')
    run_mor_method_param(so_sys.to_lti(), r, w, mus, MTReductor, 'MT')


if __name__ == "__main__":
    run(main)
