# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla


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

def msd_example(n=6, m=2, m_i=4, k_i=4, c_i=1, as_lti=False):
    """Mass-spring-damper model as (port-Hamiltonian) linear time-invariant system.

    Taken from :cite:`GPBV12`.

    Parameters
    ----------
    n
        The order of the model.
    m
        The number or inputs and outputs of the model.
    m_i
        The weight of the masses.
    k_i
        The stiffness of the springs.
    c_i
        The amount of damping.
    as_lti
        If `True`, the matrices of the standard linear time-invariant system are returned.
        Otherwise, the matrices of the port-Hamiltonian linear time-invariant system are returned.

    Returns
    -------
    A
        The LTI |NumPy array| A, if `as_lti` is `True`.
    B
        The LTI |NumPy array| B, if `as_lti` is `True`.
    C
        The LTI |NumPy array| C, if `as_lti` is `True`.
    D
        The LTI |NumPy array| D, if `as_lti` is `True`.
    J
        The pH |NumPy array| J, if `as_lti` is `False`.
    R
        The pH |NumPy array| R, if `as_lti` is `False`.
    G
        The pH |NumPy array| G, if `as_lti` is `False`.
    P
        The pH |NumPy array| P, if `as_lti` is `False`.
    S
        The pH |NumPy array| S, if `as_lti` is `False`.
    N
        The pH |NumPy array| N, if `as_lti` is `False`.
    E
        The LTI |NumPy array| E, if `as_lti` is `True`, or
        the pH |NumPy array| E, if `as_lti` is `False`.
    """
    assert n % 2 == 0
    n //= 2

    A = np.array(
        [[0, 1 / m_i, 0, 0, 0, 0], [-k_i, -c_i / m_i, k_i, 0, 0, 0],
         [0, 0, 0, 1 / m_i, 0, 0], [k_i, 0, -2 * k_i, -c_i / m_i, k_i, 0],
         [0, 0, 0, 0, 0, 1 / m_i], [0, 0, k_i, 0, -2 * k_i, -c_i / m_i]])

    if m == 2:
        B = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]).T
        C = np.array([[0, 1 / m_i, 0, 0, 0, 0], [0, 0, 0, 1 / m_i, 0, 0]])
    elif m == 1:
        B = np.array([[0, 1, 0, 0, 0, 0]]).T
        C = np.array([[0, 1 / m_i, 0, 0, 0, 0]])
    else:
        assert False

    J_i = np.array([[0, 1], [-1, 0]])
    J = np.kron(np.eye(3), J_i)
    R_i = np.array([[0, 0], [0, c_i]])
    R = np.kron(np.eye(3), R_i)

    for i in range(4, n + 1):
        B = np.vstack((B, np.zeros((2, m))))
        C = np.hstack((C, np.zeros((m, 2))))

        J = np.block([
            [J, np.zeros(((i - 1) * 2, 2))],
            [np.zeros((2, (i - 1) * 2)), J_i]
        ])

        R = np.block([
            [R, np.zeros(((i - 1) * 2, 2))],
            [np.zeros((2, (i - 1) * 2)), R_i]
        ])

        A = np.block([
            [A, np.zeros(((i - 1) * 2, 2))],
            [np.zeros((2, i * 2))]
        ])

        A[2 * i - 2, 2 * i - 2] = 0
        A[2 * i - 1, 2 * i - 1] = -c_i / m_i
        A[2 * i - 3, 2 * i - 2] = k_i
        A[2 * i - 2, 2 * i - 1] = 1 / m_i
        A[2 * i - 2, 2 * i - 3] = 0
        A[2 * i - 1, 2 * i - 2] = -2 * k_i
        A[2 * i - 1, 2 * i - 4] = k_i

    Q = spla.solve(J - R, A)
    G = B
    P = np.zeros(G.shape)
    D = np.zeros((m, m))
    E = np.eye(2 * n)
    S = (D + D.T) / 2
    N = -(D - D.T) / 2

    if as_lti:
        return A, B, C, D, E

    return J, R, G, P, S, N, E, Q
