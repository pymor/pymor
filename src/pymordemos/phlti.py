#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from matplotlib import pyplot as plt
from typer import Argument, run

from pymor.models.iosys import LTIModel, PHLTIModel
from pymor.reductors.bt import PRBTReductor
from pymor.reductors.h2 import IRKAReductor
from pymor.reductors.ph.ph_irka import PHIRKAReductor


def msd(n=6, m=2, m_i=4, k_i=4, c_i=1, as_lti=False):
    """Mass-spring-damper model as (port-Hamiltonian) linear time-invariant system.

    Taken from :cite:`GPBV12`.

    Parameters
    ----------
    n
        The order of the model.
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

    Q = np.linalg.solve(J - R, A)
    G = B
    P = np.zeros(G.shape)
    D = np.zeros((m, m))
    E = np.eye(2 * n)
    S = (D + D.T) / 2
    N = -(D - D.T) / 2

    if as_lti:
        return A, B, C, D, E

    return J, R, G, P, S, N, E, Q


def main(
        n: int = Argument(100, help='Order of the mass-spring-damper system.'),
        m: int = Argument(2, help='Number of inputs and outputs of the mass-spring-damper system.'),
        reduced_order: int = Argument(0, help='The reduced order if positive. Otherwise, a range of values is used.'),
):
    A, B, C, D, E = msd(n, m, as_lti=True)
    lti = LTIModel.from_matrices(A, B, C, D, E)
    phlti = PHLTIModel.from_passive_LTIModel(lti)

    J, R, G, P, S, N, E, Q = msd(n, m)
    fom = PHLTIModel.from_matrices(J, R, G, Q=Q)

    print(f'Error: {(fom - phlti).h2_norm():.2e}')

    prbt = PRBTReductor(fom)
    irka = IRKAReductor(fom)
    phirka = PHIRKAReductor(fom)

    reductors = {'PRBT': prbt, 'IRKA': irka, 'pH-IRKA': phirka}
    markers = {'PRBT': 'x', 'IRKA': 'o', 'pH-IRKA': 's'}

    if reduced_order > 0:
        reduced_order = [reduced_order]
    else:
        reduced_order = range(2, 22, 2)
    h2_errors = np.zeros((len(reductors), len(reduced_order)))

    for i, reductor in enumerate(reductors.values()):
        for j, r in enumerate(reduced_order):
            rom = reductor.reduce(r)
            if not isinstance(rom, PHLTIModel):
                rom = PHLTIModel.from_passive_LTIModel(rom)
            h2_errors[i, j] = (rom - fom).h2_norm() / fom.h2_norm()

    fig, ax = plt.subplots()
    for i, reductor_name in enumerate(reductors):
        ax.semilogy(reduced_order, h2_errors[i], label=reductor_name, marker=markers[reductor_name])
    ax.set_xlabel('Reduced order $r$')
    ax.set_ylabel('Relative $\\mathcal{H}_2$-error')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    run(main)
