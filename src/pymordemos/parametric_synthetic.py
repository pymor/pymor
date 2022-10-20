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
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.bt import BTReductor
from pymor.reductors.h2 import IRKAReductor
from pymordemos.parametric_heat import fom_properties_param, run_mor_method_param


def main(
        n: int = Argument(100, help='Order of the FOM.'),
        r: int = Argument(10, help='Order of the ROMs.'),
):
    """Synthetic parametric demo.

    See the `MOR Wiki page <http://modelreduction.org/index.php/Synthetic_parametric_model>`_.
    """
    set_log_levels({
        'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING',
        'pymor.algorithms.lradi.solve_lyap_lrcf': 'WARNING',
        'pymor.reductors.basic.LTIPGReductor': 'WARNING',
    })
    plt.rcParams['axes.grid'] = True

    # Model
    # set coefficients
    a = -np.linspace(1e1, 1e3, n // 2)
    b = np.linspace(1e1, 1e3, n // 2)
    c = np.ones(n // 2)
    d = np.zeros(n // 2)

    # build 2x2 submatrices
    aa = np.empty(n)
    aa[::2] = a
    aa[1::2] = a
    bb = np.zeros(n)
    bb[::2] = b

    # set up system matrices
    Amu = sps.diags(aa, format='csc')
    A0 = sps.diags([bb, -bb], [1, -1], shape=(n, n), format='csc')
    B = np.zeros((n, 1))
    B[::2, 0] = 2
    C = np.empty((1, n))
    C[0, ::2] = c
    C[0, 1::2] = d

    # form operators
    A0 = NumpyMatrixOperator(A0)
    Amu = NumpyMatrixOperator(Amu)
    B = NumpyMatrixOperator(B)
    C = NumpyMatrixOperator(C)
    A = A0 + Amu * ProjectionParameterFunctional('mu')

    # form a model
    lti = LTIModel(A, B, C)

    mus = [1/50, 1/20, 1/10, 1/5, 1/2, 1]
    w = (10**0.5, 10**3.5)
    fom_properties_param(lti, w, mus)

    # Model order reduction
    run_mor_method_param(lti, r, w, mus, BTReductor, 'BT')
    run_mor_method_param(lti, r, w, mus, IRKAReductor, 'IRKA')


if __name__ == "__main__":
    run(main)
