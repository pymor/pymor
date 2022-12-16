#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt
from typer import run

from pymor.core.logger import set_log_levels
from pymor.models.iosys import LTIModel
from pymor.reductors.bt import BTReductor
from pymor.reductors.era import ERAReductor
from pymor.tools.random import get_rng, new_rng


def example_system(order, sampling_time=1):
    """Construct an example system.

    The system is adapted from Section III-C of https://ieeexplore.ieee.org/document/508900.

    Parameters
    ----------
    order
        The order of the constructed LTI system.
    sampling_time
        The sampling time of the LTI system.

    Returns
    -------
    sys
        The constructed |LTIModel|.
    """
    rng = get_rng()
    A = np.eye(order) * 1e-3
    B = rng.random((order, 1)) / 2
    C = rng.random((1, order)) / 2

    A[:2, :2] = np.array([[0.8876, 0.4494], [-0.4494, 0.7978]])
    A[2:4, 2:4] = np.array([[-0.6129, 0.0645], [-6.4516, -0.7419]])
    B[:4, :] = np.array([[0.2247], [0.8989], [0.0323], [0.1290]])
    C[:, :4] = np.array([0.4719, 0.1124, 9.6774, 1.6129])
    D = np.array([0.9626])
    return LTIModel.from_matrices(A, B, C, D=D, sampling_time=sampling_time)


def compute_markov_parameters(sys, n=100):
    """Compute the Markov parameters of a system.

    Parameters
    ----------
    sys
        The |LTIModel| for which the Markov parameters are computed.
    n
        The number of Markov parameters that are calculated.

    Returns
    -------
    mp
        |NumpyArray| of shape (n, sys.dim_ouputs, sys.dim_inputs).
    """
    mp = np.zeros((n, sys.dim_output, sys.dim_input))
    A, B, C, *_ = sys.to_matrices()
    x = B
    for i in range(n):
        mp[i] = np.squeeze(C @ x)
        x = A @ x
    return mp


def main():
    set_log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING'})

    sampling_time = 0.1
    w = np.geomspace(1e-2, 1, 100) * np.pi
    with new_rng(0):
        fom = example_system(10, sampling_time=sampling_time)

    mp = compute_markov_parameters(fom, n=100)

    era = ERAReductor(mp, sampling_time=sampling_time, feedthrough=fom.D)
    bt = BTReductor(fom)

    era_rom = era.reduce(4)
    bt_rom = bt.reduce(4)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax = axs[0]
    fom.transfer_function.mag_plot(w, ax=ax, label='FOM', dB=True)
    era_rom.transfer_function.mag_plot(w, ax=ax, label='ERA ROM', dB=True, linestyle='dashed')
    bt_rom.transfer_function.mag_plot(w, ax=ax, label='BT ROM', dB=True, linestyle='dotted')
    ax.set_title(r'Transfer function magnitude ($r=4$)')
    ax.legend()

    ax = axs[1]
    (fom-era_rom).transfer_function.mag_plot(w, ax=ax, label='ERA', dB=True)
    (fom-bt_rom).transfer_function.mag_plot(w, ax=ax, label='BT', dB=True, linestyle='dashed')
    ax.set_title(r'Error magnitude ($r=4$)')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    run(main)
