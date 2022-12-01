#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from matplotlib import pyplot as plt
from typer import run, Option

from pymor.models.iosys import PHLTIModel, LTIModel


def msd(n=6, m_i=4, k_i=4, c_i=1, as_lti=False):
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
        If `True`, the matrices of the standard linear time-invariant system are returned,
        otherwise the matrices of the port-hamiltonian linear time-invariant system are returned.

    Returns
    -------
    A
        The lti |NumPy array| A, if `as_lti` is `True`.
    B
        The lti |NumPy array| B, if `as_lti` is `True`.
    C
        The lti |NumPy array| C, if `as_lti` is `True`.
    D
        The lti |NumPy array| D, if `as_lti` is `True`.
    J
        The ph |NumPy array| J, if `as_lti` is `False`.
    R
        The ph |NumPy array| R, if `as_lti` is `False`.
    G
        The ph |NumPy array| G, if `as_lti` is `False`.
    P
        The ph |NumPy array| P, if `as_lti` is `False`.
    S
        The ph |NumPy array| S, if `as_lti` is `False`.
    N
        The ph |NumPy array| N, if `as_lti` is `False`.
    E
        The lti |NumPy array| E, if `as_lti` is `True`, or
        the ph |NumPy array| E, if `as_lti` is `False`.
    """
    n = int(n / 2)
    m = 1

    A = np.array(
        [[0, 1 / m_i, 0, 0, 0, 0], [-k_i, -c_i / m_i, k_i, 0, 0, 0],
         [0, 0, 0, 1 / m_i, 0, 0], [k_i, 0, -2 * k_i, -c_i / m_i, k_i, 0],
         [0, 0, 0, 0, 0, 1 / m_i], [0, 0, k_i, 0, -2 * k_i, -c_i / m_i]])

    B = np.array([[0, 1, 0, 0, 0, 0]]).T
    C = np.array([[0, 1 / m_i, 0, 0, 0, 0]])

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

    # Shift Q on LHS
    E = Q.T @ E
    J = Q.T @ J @ Q
    R = Q.T @ R @ Q
    G = Q.T @ G
    P = Q.T @ P

    return J, R, G, P, S, N, E


def main(
        n: int = Option(10, help='Order of the Mass-Spring-Damper system.')
):
    A, B, C, D, E = msd(n, as_lti=True)
    lti = LTIModel.from_matrices(A, B, C, D, E)

    J, R, G, P, S, N, E = msd(n)

    phlti = PHLTIModel.from_matrices(J, R, G, P, S, N, E)
    print(phlti)

    # Magnitude plot
    w = np.logspace(-2, 8, 300)
    fig, ax = plt.subplots()
    _ = lti.transfer_function.mag_plot(w, ax=ax, label='LTI')
    _ = phlti.transfer_function.mag_plot(w, ax=ax, ls='--', label='PH')
    ax.legend()
    plt.show()

    # Poles
    poles = phlti.poles()
    poles_lti = lti.poles()

    fig, ax = plt.subplots()
    ax.scatter(poles_lti.real, poles_lti.imag, marker='x', label='LTI')
    ax.scatter(poles.real, poles.imag, marker='o', facecolors='none', edgecolor='orange', label='PH')
    ax.set_title('Poles')
    ax.legend()
    ax.set(xlabel=r'Re($\lambda$)', ylabel=r'Im($\lambda$)')
    plt.show()

    e = phlti - lti
    e.transfer_function.mag_plot(w)
    plt.show()


if __name__ == '__main__':
    run(main)
