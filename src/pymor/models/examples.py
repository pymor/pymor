# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

def thermal_block_example():
    """Return 2x2 thermal block example.

    Returns
    -------
    fom
        Thermal block problem as a |StationaryModel|.
    """
    from pymor.analyticalproblems.thermalblock import thermal_block_problem
    from pymor.discretizers.builtin import discretize_stationary_cg

    p = thermal_block_problem((2, 2))
    fom, _ = discretize_stationary_cg(p, diameter=1/100)
    return fom

def penzl_example():
    """Return Penzl's example.

    Returns
    -------
    fom
        Penzl's FOM example as an |LTIModel|.
    """
    import numpy as np
    import scipy.sparse as sps

    from pymor.models.iosys import LTIModel

    n = 1006
    A1 = np.array([[-1, 100], [-100, -1]])
    A2 = np.array([[-1, 200], [-200, -1]])
    A3 = np.array([[-1, 400], [-400, -1]])
    A4 = sps.diags(np.arange(-1, -n + 5, -1))
    A = sps.block_diag((A1, A2, A3, A4))
    B = np.ones((n, 1))
    B[:6] = 10
    C = B.T
    fom = LTIModel.from_matrices(A, B, C)

    return fom
