#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""String equation example"""

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from pymor.core.config import config
from pymor.discretizations.iosys import SecondOrderSystem
from pymor.reductors.bt import BTReductor
from pymor.reductors.h2 import IRKAReductor
from pymor.reductors.sobt import (SOBTpReductor, SOBTvReductor, SOBTpvReductor, SOBTvpReductor,
                                  SOBTfvReductor, SOBTReductor)
from pymor.reductors.sor_irka import SOR_IRKAReductor

import logging
logging.getLogger('pymor.algorithms.gram_schmidt.gram_schmidt').setLevel(logging.ERROR)


def compute_hinf_norm(message, sys):
    if config.HAVE_SLYCOT:
        print(message.format(sys.hinf_norm()))
    else:
        print('H_inf-norm calculation is skipped due to missing slycot.')


if __name__ == '__main__':
    # Assemble matrices
    n2 = 50
    n = 2 * n2 - 1  # dimension of the system

    d = 10  # damping
    k = 1   # stiffness

    M = sps.eye(n, format='csc')

    E = d * sps.eye(n, format='csc')

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
    so_sys = SecondOrderSystem.from_matrices(M, E, K, B, Cp)

    print('n = {}'.format(so_sys.n))
    print('m = {}'.format(so_sys.m))
    print('p = {}'.format(so_sys.p))

    poles = so_sys.poles()
    fig, ax = plt.subplots()
    ax.plot(poles.real, poles.imag, '.')
    ax.set_title('System poles')
    plt.show()

    w = np.logspace(-1, 3, 1000)
    fig, ax = plt.subplots()
    so_sys.mag_plot(w, ax=ax)
    ax.set_title('Bode plot of the full model')
    plt.show()

    psv = so_sys.psv()
    vsv = so_sys.vsv()
    pvsv = so_sys.pvsv()
    vpsv = so_sys.vpsv()
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    ax[0, 0].semilogy(range(1, len(psv) + 1), psv, '.-')
    ax[0, 0].set_title('Position singular values')
    ax[0, 1].semilogy(range(1, len(vsv) + 1), vsv, '.-')
    ax[0, 1].set_title('Velocity singular values')
    ax[1, 0].semilogy(range(1, len(pvsv) + 1), pvsv, '.-')
    ax[1, 0].set_title('Position-velocity singular values')
    ax[1, 1].semilogy(range(1, len(vpsv) + 1), vpsv, '.-')
    ax[1, 1].set_title('Velocity-position singular values')
    plt.show()

    print('H_2-norm of the full model:    {:e}'.format(so_sys.h2_norm()))
    compute_hinf_norm('H_inf-norm of the full model:  {:e}', so_sys)
    print('Hankel-norm of the full model: {:e}'.format(so_sys.hankel_norm()))

    # Position Second-Order Balanced Truncation (SOBTp)
    r = 10
    sobtp_reductor = SOBTpReductor(so_sys)
    rom_sobtp = sobtp_reductor.reduce(r)

    poles_rom_sobtp = rom_sobtp.poles()
    fig, ax = plt.subplots()
    ax.plot(poles_rom_sobtp.real, poles_rom_sobtp.imag, '.')
    ax.set_title("SOBTp reduced model's poles")
    plt.show()

    err_sobtp = so_sys - rom_sobtp
    print('H_2-error for the SOBTp ROM:    {:e}'.format(err_sobtp.h2_norm()))
    compute_hinf_norm('H_inf-error for the SOBTp ROM:  {:e}', err_sobtp)
    print('Hankel-error for the SOBTp ROM: {:e}'.format(err_sobtp.hankel_norm()))

    fig, ax = plt.subplots()
    so_sys.mag_plot(w, ax=ax)
    rom_sobtp.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title('Bode plot of the full and SOBTp reduced model')
    plt.show()

    fig, ax = plt.subplots()
    err_sobtp.mag_plot(w, ax=ax)
    ax.set_title('Bode plot of the SOBTp error system')
    plt.show()

    # Velocity Second-Order Balanced Truncation (SOBTv)
    r = 10
    sobtv_reductor = SOBTvReductor(so_sys)
    rom_sobtv = sobtv_reductor.reduce(r)

    poles_rom_sobtv = rom_sobtv.poles()
    fig, ax = plt.subplots()
    ax.plot(poles_rom_sobtv.real, poles_rom_sobtv.imag, '.')
    ax.set_title("SOBTv reduced model's poles")
    plt.show()

    err_sobtv = so_sys - rom_sobtv
    print('H_2-error for the SOBTv ROM:    {:e}'.format(err_sobtv.h2_norm()))
    compute_hinf_norm('H_inf-error for the SOBTv ROM:  {:e}', err_sobtv)
    print('Hankel-error for the SOBTv ROM: {:e}'.format(err_sobtv.hankel_norm()))

    fig, ax = plt.subplots()
    so_sys.mag_plot(w, ax=ax)
    rom_sobtv.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title('Bode plot of the full and SOBTv reduced model')
    plt.show()

    fig, ax = plt.subplots()
    err_sobtv.mag_plot(w, ax=ax)
    ax.set_title('Bode plot of the SOBTv error system')
    plt.show()

    # Position-Velocity Second-Order Balanced Truncation (SOBTpv)
    r = 10
    sobtpv_reductor = SOBTpvReductor(so_sys)
    rom_sobtpv = sobtpv_reductor.reduce(r)

    poles_rom_sobtpv = rom_sobtpv.poles()
    fig, ax = plt.subplots()
    ax.plot(poles_rom_sobtpv.real, poles_rom_sobtpv.imag, '.')
    ax.set_title("SOBTpv reduced model's poles")
    plt.show()

    err_sobtpv = so_sys - rom_sobtpv
    print('H_2-error for the SOBTpv ROM:    {:e}'.format(err_sobtpv.h2_norm()))
    compute_hinf_norm('H_inf-error for the SOBTpv ROM:  {:e}', err_sobtpv)
    print('Hankel-error for the SOBTpv ROM: {:e}'.format(err_sobtpv.hankel_norm()))

    fig, ax = plt.subplots()
    so_sys.mag_plot(w, ax=ax)
    rom_sobtpv.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title('Bode plot of the full and SOBTpv reduced model')
    plt.show()

    fig, ax = plt.subplots()
    err_sobtpv.mag_plot(w, ax=ax)
    ax.set_title('Bode plot of the SOBTpv error system')
    plt.show()

    # Velocity-Position Second-Order Balanced Truncation (SOBTvp)
    r = 10
    sobtvp_reductor = SOBTvpReductor(so_sys)
    rom_sobtvp = sobtvp_reductor.reduce(r)

    poles_rom_sobtvp = rom_sobtvp.poles()
    fig, ax = plt.subplots()
    ax.plot(poles_rom_sobtvp.real, poles_rom_sobtvp.imag, '.')
    ax.set_title("SOBTvp reduced model's poles")
    plt.show()

    err_sobtvp = so_sys - rom_sobtvp
    print('H_2-error for the SOBTvp ROM:    {:e}'.format(err_sobtvp.h2_norm()))
    compute_hinf_norm('H_inf-error for the SOBTvp ROM:  {:e}', err_sobtvp)
    print('Hankel-error for the SOBTvp ROM: {:e}'.format(err_sobtvp.hankel_norm()))

    fig, ax = plt.subplots()
    so_sys.mag_plot(w, ax=ax)
    rom_sobtvp.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title('Bode plot of the full and SOBTvp reduced model')
    plt.show()

    fig, ax = plt.subplots()
    err_sobtvp.mag_plot(w, ax=ax)
    ax.set_title('Bode plot of the SOBTvp error system')
    plt.show()

    # Free-Velocity Second-Order Balanced Truncation (SOBTfv)
    r = 10
    sobtfv_reductor = SOBTfvReductor(so_sys)
    rom_sobtfv = sobtfv_reductor.reduce(r)

    poles_rom_sobtfv = rom_sobtfv.poles()
    fig, ax = plt.subplots()
    ax.plot(poles_rom_sobtfv.real, poles_rom_sobtfv.imag, '.')
    ax.set_title("SOBTfv reduced model's poles")
    plt.show()

    err_sobtfv = so_sys - rom_sobtfv
    print('H_2-error for the SOBTfv ROM:    {:e}'.format(err_sobtfv.h2_norm()))
    compute_hinf_norm('H_inf-error for the SOBTfv ROM:  {:e}', err_sobtfv)
    print('Hankel-error for the SOBTfv ROM: {:e}'.format(err_sobtfv.hankel_norm()))

    fig, ax = plt.subplots()
    so_sys.mag_plot(w, ax=ax)
    rom_sobtfv.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title('Bode plot of the full and SOBTfv reduced model')
    plt.show()

    fig, ax = plt.subplots()
    err_sobtfv.mag_plot(w, ax=ax)
    ax.set_title('Bode plot of the SOBTfv error system')
    plt.show()

    # Second-Order Balanced Truncation (SOBT)
    r = 10
    sobt_reductor = SOBTReductor(so_sys)
    rom_sobt = sobt_reductor.reduce(r)

    poles_rom_sobt = rom_sobt.poles()
    fig, ax = plt.subplots()
    ax.plot(poles_rom_sobt.real, poles_rom_sobt.imag, '.')
    ax.set_title("SOBT reduced model's poles")
    plt.show()

    err_sobt = so_sys - rom_sobt
    print('H_2-error for the SOBT ROM:    {:e}'.format(err_sobt.h2_norm()))
    compute_hinf_norm('H_inf-error for the SOBT ROM:  {:e}', err_sobt)
    print('Hankel-error for the SOBT ROM: {:e}'.format(err_sobt.hankel_norm()))

    fig, ax = plt.subplots()
    so_sys.mag_plot(w, ax=ax)
    rom_sobt.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title('Bode plot of the full and SOBT reduced model')
    plt.show()

    fig, ax = plt.subplots()
    err_sobt.mag_plot(w, ax=ax)
    ax.set_title('Bode plot of the SOBT error system')
    plt.show()

    # Balanced Truncation (BT)
    r = 10
    bt_reductor = BTReductor(so_sys.to_lti())
    rom_bt = bt_reductor.reduce(r)

    poles_rom_bt = rom_bt.poles()
    fig, ax = plt.subplots()
    ax.plot(poles_rom_bt.real, poles_rom_bt.imag, '.')
    ax.set_title("BT reduced model's poles")
    plt.show()

    err_bt = so_sys.to_lti() - rom_bt
    print('H_2-error for the BT ROM:    {:e}'.format(err_bt.h2_norm()))
    compute_hinf_norm('H_inf-error for the BT ROM:  {:e}', err_bt)
    print('Hankel-error for the BT ROM: {:e}'.format(err_bt.hankel_norm()))

    fig, ax = plt.subplots()
    so_sys.mag_plot(w, ax=ax)
    rom_bt.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title('Bode plot of the full and BT reduced model')
    plt.show()

    fig, ax = plt.subplots()
    err_bt.mag_plot(w, ax=ax)
    ax.set_title('Bode plot of the BT error system')
    plt.show()

    # Iterative Rational Krylov Algorithm (IRKA)
    r = 10
    irka_reductor = IRKAReductor(so_sys.to_lti())
    rom_irka = irka_reductor.reduce(r, num_prev=2, conv_crit='h2')

    fig, ax = plt.subplots()
    ax.semilogy(irka_reductor.dist, '.-')
    ax.set_title('IRKA convergence criterion')
    plt.show()

    poles_rom_irka = rom_irka.poles()
    fig, ax = plt.subplots()
    ax.plot(poles_rom_irka.real, poles_rom_irka.imag, '.')
    ax.set_title("IRKA reduced model's poles")
    plt.show()

    err_irka = so_sys.to_lti() - rom_irka
    print('H_2-error for the IRKA ROM:    {:e}'.format(err_irka.h2_norm()))
    compute_hinf_norm('H_inf-error for the IRKA ROM:  {:e}', err_irka)
    print('Hankel-error for the IRKA ROM: {:e}'.format(err_irka.hankel_norm()))

    fig, ax = plt.subplots()
    so_sys.mag_plot(w, ax=ax)
    rom_irka.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title('Bode plot of the full and IRKA reduced model')
    plt.show()

    fig, ax = plt.subplots()
    err_irka.mag_plot(w, ax=ax)
    ax.set_title('Bode plot of the IRKA error system')
    plt.show()

    # Second-Order Iterative Rational Krylov Algorithm (SOR-IRKA)
    r = 10
    sor_irka_reductor = SOR_IRKAReductor(so_sys)
    rom_sor_irka = sor_irka_reductor.reduce(r, num_prev=2, maxit=5)

    fig, ax = plt.subplots()
    ax.semilogy(sor_irka_reductor.dist, '.-')
    ax.set_title('SOR-IRKA convergence criterion')
    plt.show()

    poles_rom_sor_irka = rom_sor_irka.poles()
    fig, ax = plt.subplots()
    ax.plot(poles_rom_sor_irka.real, poles_rom_sor_irka.imag, '.')
    ax.set_title("SOR-IRKA reduced model's poles")
    plt.show()

    err_sor_irka = so_sys - rom_sor_irka
    print('H_2-error for the SOR-IRKA ROM:    {:e}'.format(err_sor_irka.h2_norm()))
    compute_hinf_norm('H_inf-error for the SOR-IRKA ROM:  {:e}', err_sor_irka)
    print('Hankel-error for the SOR-IRKA ROM: {:e}'.format(err_sor_irka.hankel_norm()))

    fig, ax = plt.subplots()
    so_sys.mag_plot(w, ax=ax)
    rom_sor_irka.mag_plot(w, ax=ax, linestyle='dashed')
    ax.set_title('Bode plot of the full and SOR-IRKA reduced model')
    plt.show()

    fig, ax = plt.subplots()
    err_sor_irka.mag_plot(w, ax=ax)
    ax.set_title('Bode plot of the SOR-IRKA error system')
    plt.show()
