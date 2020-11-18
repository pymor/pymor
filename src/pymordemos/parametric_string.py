#!/usr/bin/env python
# coding: utf-8
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# # Parametric string equation example

# ## Import modules

# In[ ]:


import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from pymor.basic import *
from pymor.core.config import config

from pymor.core.logger import set_log_levels


def run_demo():
    set_log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING'})


    # ## Assemble $M$, $D$, $K$, $B$, $C_p$

    # In[ ]:


    n2 = 50
    n = 2 * n2 - 1  # dimension of the system

    k = 0.01   # stiffness

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


    # ## Second-order system

    # In[ ]:


    Mop = NumpyMatrixOperator(M)
    Eop = NumpyMatrixOperator(E) * ProjectionParameterFunctional('damping')
    Kop = NumpyMatrixOperator(K)
    Bop = NumpyMatrixOperator(B)
    Cpop = NumpyMatrixOperator(Cp)


    # In[ ]:


    so_sys = SecondOrderModel(Mop, Eop, Kop, Bop, Cpop)


    # In[ ]:


    print(f'order of the model = {so_sys.order}')
    print(f'number of inputs   = {so_sys.dim_input}')
    print(f'number of outputs  = {so_sys.dim_output}')


    # In[ ]:


    mu_list = [1, 5, 10]


    # In[ ]:


    fig, ax = plt.subplots()

    for mu in mu_list:
        poles = so_sys.poles(mu=mu)
        ax.plot(poles.real, poles.imag, '.', label=fr'$\mu = {mu}$')
    ax.set_title('System poles')
    ax.legend()
    plt.show()


    # In[ ]:


    w = np.logspace(-3, 2, 200)

    fig, ax = plt.subplots()
    for mu in mu_list:
        so_sys.mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of the full model')
    ax.legend()
    plt.show()


    # In[ ]:


    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for mu in mu_list:
        psv = so_sys.psv(mu=mu)
        vsv = so_sys.vsv(mu=mu)
        pvsv = so_sys.pvsv(mu=mu)
        vpsv = so_sys.vpsv(mu=mu)
        ax[0, 0].semilogy(range(1, len(psv) + 1), psv, '.-', label=fr'$\mu = {mu}$')
        ax[0, 1].semilogy(range(1, len(vsv) + 1), vsv, '.-')
        ax[1, 0].semilogy(range(1, len(pvsv) + 1), pvsv, '.-')
        ax[1, 1].semilogy(range(1, len(vpsv) + 1), vpsv, '.-')
    ax[0, 0].set_title('Position singular values')
    ax[0, 1].set_title('Velocity singular values')
    ax[1, 0].set_title('Position-velocity singular values')
    ax[1, 1].set_title('Velocity-position singular values')
    fig.legend(loc='upper center', ncol=len(mu_list))
    plt.show()


    # In[ ]:


    for mu in mu_list:
        print(f'mu = {mu}:')
        print(f'    H_2-norm of the full model:    {so_sys.h2_norm(mu=mu):e}')
        if config.HAVE_SLYCOT:
            print(f'    H_inf-norm of the full model:  {so_sys.hinf_norm(mu=mu):e}')
        print(f'    Hankel-norm of the full model: {so_sys.hankel_norm(mu=mu):e}')


    # ## Position Second-Order Balanced Truncation (SOBTp)

    # In[ ]:


    r = 5
    roms_sobtp = []
    for mu in mu_list:
        sobtp_reductor = SOBTpReductor(so_sys, mu=mu)
        rom_sobtp = sobtp_reductor.reduce(r)
        roms_sobtp.append(rom_sobtp)


    # In[ ]:


    fig, ax = plt.subplots()
    for rom_sobtp in roms_sobtp:
        poles_rom_sobtp = rom_sobtp.poles()
        ax.plot(poles_rom_sobtp.real, poles_rom_sobtp.imag, '.', label=fr'$\mu = {mu}$')
    ax.set_title("SOBTp reduced model's poles")
    plt.show()


    # In[ ]:


    for mu, rom_sobtp in zip(mu_list, roms_sobtp):
        err_sobtp = so_sys - rom_sobtp
        print(f'mu = {mu}')
        print(f'    SOBTp relative H_2-error:    {err_sobtp.h2_norm(mu=mu) / so_sys.h2_norm(mu=mu):e}')
        if config.HAVE_SLYCOT:
            print(f'    SOBTp relative H_inf-error:  {err_sobtp.hinf_norm(mu=mu) / so_sys.hinf_norm(mu=mu):e}')
        print(f'    SOBTp relative Hankel-error: {err_sobtp.hankel_norm(mu=mu) / so_sys.hankel_norm(mu=mu):e}')


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_sobtp in zip(mu_list, roms_sobtp):
        rom_sobtp.mag_plot(w, ax=ax, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of SOBTp reduced models')
    ax.legend()
    plt.show()


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_sobtp in zip(mu_list, roms_sobtp):
        (so_sys - rom_sobtp).mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of the SOBTp error system')
    ax.legend()
    plt.show()


    # ## Velocity Second-Order Balanced Truncation (SOBTv)

    # In[ ]:


    r = 5
    roms_sobtv = []
    for mu in mu_list:
        sobtv_reductor = SOBTvReductor(so_sys, mu=mu)
        rom_sobtv = sobtv_reductor.reduce(r)
        roms_sobtv.append(rom_sobtv)


    # In[ ]:


    fig, ax = plt.subplots()
    for rom_sobtv in roms_sobtv:
        poles_rom_sobtv = rom_sobtv.poles()
        ax.plot(poles_rom_sobtv.real, poles_rom_sobtv.imag, '.', label=fr'$\mu = {mu}$')
    ax.set_title("SOBTv reduced model's poles")
    plt.show()


    # In[ ]:


    for mu, rom_sobtv in zip(mu_list, roms_sobtv):
        err_sobtv = so_sys - rom_sobtv
        print(f'mu = {mu}')
        print(f'    SOBTv relative H_2-error:    {err_sobtv.h2_norm(mu=mu) / so_sys.h2_norm(mu=mu):e}')
        if config.HAVE_SLYCOT:
            print(f'    SOBTv relative H_inf-error:  {err_sobtv.hinf_norm(mu=mu) / so_sys.hinf_norm(mu=mu):e}')
        print(f'    SOBTv relative Hankel-error: {err_sobtv.hankel_norm(mu=mu) / so_sys.hankel_norm(mu=mu):e}')


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_sobtv in zip(mu_list, roms_sobtv):
        rom_sobtv.mag_plot(w, ax=ax, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of SOBTv reduced models')
    ax.legend()
    plt.show()


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_sobtv in zip(mu_list, roms_sobtv):
        (so_sys - rom_sobtv).mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of the SOBTv error system')
    ax.legend()
    plt.show()


    # ## Position-Velocity Second-Order Balanced Truncation (SOBTpv)

    # In[ ]:


    r = 5
    roms_sobtpv = []
    for mu in mu_list:
        sobtpv_reductor = SOBTpvReductor(so_sys, mu=mu)
        rom_sobtpv = sobtpv_reductor.reduce(r)
        roms_sobtpv.append(rom_sobtpv)


    # In[ ]:


    fig, ax = plt.subplots()
    for rom_sobtpv in roms_sobtpv:
        poles_rom_sobtpv = rom_sobtpv.poles()
        ax.plot(poles_rom_sobtpv.real, poles_rom_sobtpv.imag, '.', label=fr'$\mu = {mu}$')
    ax.set_title("SOBTpv reduced model's poles")
    plt.show()


    # In[ ]:


    for mu, rom_sobtpv in zip(mu_list, roms_sobtpv):
        err_sobtpv = so_sys - rom_sobtpv
        print(f'mu = {mu}')
        print(f'    SOBTpv relative H_2-error:    {err_sobtpv.h2_norm(mu=mu) / so_sys.h2_norm(mu=mu):e}')
        if config.HAVE_SLYCOT:
            print(f'    SOBTpv relative H_inf-error:  {err_sobtpv.hinf_norm(mu=mu) / so_sys.hinf_norm(mu=mu):e}')
        print(f'    SOBTpv relative Hankel-error: {err_sobtpv.hankel_norm(mu=mu) / so_sys.hankel_norm(mu=mu):e}')


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_sobtpv in zip(mu_list, roms_sobtpv):
        rom_sobtpv.mag_plot(w, ax=ax, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of SOBTpv reduced models')
    ax.legend()
    plt.show()


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_sobtpv in zip(mu_list, roms_sobtpv):
        (so_sys - rom_sobtpv).mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of the SOBTpv error system')
    ax.legend()
    plt.show()


    # ## Velocity-Position Second-Order Balanced Truncation (SOBTvp)

    # In[ ]:


    r = 5
    roms_sobtvp = []
    for mu in mu_list:
        sobtvp_reductor = SOBTvpReductor(so_sys, mu=mu)
        rom_sobtvp = sobtvp_reductor.reduce(r)
        roms_sobtvp.append(rom_sobtvp)


    # In[ ]:


    fig, ax = plt.subplots()
    for rom_sobtvp in roms_sobtvp:
        poles_rom_sobtvp = rom_sobtvp.poles()
        ax.plot(poles_rom_sobtvp.real, poles_rom_sobtvp.imag, '.', label=fr'$\mu = {mu}$')
    ax.set_title("SOBTvp reduced model's poles")
    plt.show()


    # In[ ]:


    for mu, rom_sobtvp in zip(mu_list, roms_sobtvp):
        err_sobtvp = so_sys - rom_sobtvp
        print(f'mu = {mu}')
        print(f'    SOBTvp relative H_2-error:    {err_sobtvp.h2_norm(mu=mu) / so_sys.h2_norm(mu=mu):e}')
        if config.HAVE_SLYCOT:
            print(f'    SOBTvp relative H_inf-error:  {err_sobtvp.hinf_norm(mu=mu) / so_sys.hinf_norm(mu=mu):e}')
        print(f'    SOBTvp relative Hankel-error: {err_sobtvp.hankel_norm(mu=mu) / so_sys.hankel_norm(mu=mu):e}')


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_sobtvp in zip(mu_list, roms_sobtvp):
        rom_sobtvp.mag_plot(w, ax=ax, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of SOBTvp reduced models')
    ax.legend()
    plt.show()


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_sobtvp in zip(mu_list, roms_sobtvp):
        (so_sys - rom_sobtvp).mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of the SOBTvp error system')
    ax.legend()
    plt.show()


    # ## Free-Velocity Second-Order Balanced Truncation (SOBTfv)

    # In[ ]:


    r = 5
    roms_sobtfv = []
    for mu in mu_list:
        sobtfv_reductor = SOBTfvReductor(so_sys, mu=mu)
        rom_sobtfv = sobtfv_reductor.reduce(r)
        roms_sobtfv.append(rom_sobtfv)


    # In[ ]:


    fig, ax = plt.subplots()
    for rom_sobtfv in roms_sobtfv:
        poles_rom_sobtfv = rom_sobtfv.poles()
        ax.plot(poles_rom_sobtfv.real, poles_rom_sobtfv.imag, '.', label=fr'$\mu = {mu}$')
    ax.set_title("SOBTfv reduced model's poles")
    plt.show()


    # In[ ]:


    for mu, rom_sobtfv in zip(mu_list, roms_sobtfv):
        err_sobtfv = so_sys - rom_sobtfv
        print(f'mu = {mu}')
        print(f'    SOBTfv relative H_2-error:    {err_sobtfv.h2_norm(mu=mu) / so_sys.h2_norm(mu=mu):e}')
        if config.HAVE_SLYCOT:
            print(f'    SOBTfv relative H_inf-error:  {err_sobtfv.hinf_norm(mu=mu) / so_sys.hinf_norm(mu=mu):e}')
        print(f'    SOBTfv relative Hankel-error: {err_sobtfv.hankel_norm(mu=mu) / so_sys.hankel_norm(mu=mu):e}')


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_sobtfv in zip(mu_list, roms_sobtfv):
        rom_sobtfv.mag_plot(w, ax=ax, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of SOBTfv reduced models')
    ax.legend()
    plt.show()


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_sobtfv in zip(mu_list, roms_sobtfv):
        (so_sys - rom_sobtfv).mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of the SOBTfv error system')
    ax.legend()
    plt.show()


    # ## Second-Order Balanced Truncation (SOBT)

    # In[ ]:


    r = 5
    roms_sobt = []
    for mu in mu_list:
        sobt_reductor = SOBTReductor(so_sys, mu=mu)
        rom_sobt = sobt_reductor.reduce(r)
        roms_sobt.append(rom_sobt)


    # In[ ]:


    fig, ax = plt.subplots()
    for rom_sobt in roms_sobt:
        poles_rom_sobt = rom_sobt.poles()
        ax.plot(poles_rom_sobt.real, poles_rom_sobt.imag, '.', label=fr'$\mu = {mu}$')
    ax.set_title("SOBT reduced model's poles")
    plt.show()


    # In[ ]:


    for mu, rom_sobt in zip(mu_list, roms_sobt):
        err_sobt = so_sys - rom_sobt
        print(f'mu = {mu}')
        print(f'    SOBT relative H_2-error:    {err_sobt.h2_norm(mu=mu) / so_sys.h2_norm(mu=mu):e}')
        if config.HAVE_SLYCOT:
            print(f'    SOBT relative H_inf-error:  {err_sobt.hinf_norm(mu=mu) / so_sys.hinf_norm(mu=mu):e}')
        print(f'    SOBT relative Hankel-error: {err_sobt.hankel_norm(mu=mu) / so_sys.hankel_norm(mu=mu):e}')


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_sobt in zip(mu_list, roms_sobt):
        rom_sobt.mag_plot(w, ax=ax, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of SOBT reduced models')
    ax.legend()
    plt.show()


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_sobt in zip(mu_list, roms_sobt):
        (so_sys - rom_sobt).mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of the SOBT error system')
    ax.legend()
    plt.show()


    # ## Balanced Truncation (BT)

    # In[ ]:


    r = 5
    roms_bt = []
    for mu in mu_list:
        bt_reductor = BTReductor(so_sys.to_lti(), mu=mu)
        rom_bt = bt_reductor.reduce(r)
        roms_bt.append(rom_bt)


    # In[ ]:


    fig, ax = plt.subplots()
    for rom_bt in roms_bt:
        poles_rom_bt = rom_bt.poles()
        ax.plot(poles_rom_bt.real, poles_rom_bt.imag, '.', label=fr'$\mu = {mu}$')
    ax.set_title("BT reduced model's poles")
    plt.show()


    # In[ ]:


    for mu, rom_bt in zip(mu_list, roms_bt):
        err_bt = so_sys - rom_bt
        print(f'mu = {mu}')
        print(f'    BT relative H_2-error:    {err_bt.h2_norm(mu=mu) / so_sys.h2_norm(mu=mu):e}')
        if config.HAVE_SLYCOT:
            print(f'    BT relative H_inf-error:  {err_bt.hinf_norm(mu=mu) / so_sys.hinf_norm(mu=mu):e}')
        print(f'    BT relative Hankel-error: {err_bt.hankel_norm(mu=mu) / so_sys.hankel_norm(mu=mu):e}')


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_bt in zip(mu_list, roms_bt):
        rom_bt.mag_plot(w, ax=ax, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of BT reduced models')
    ax.legend()
    plt.show()


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_bt in zip(mu_list, roms_bt):
        (so_sys - rom_bt).mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of the BT error system')
    ax.legend()
    plt.show()


    # ## Iterative Rational Krylov Algorithm (IRKA)

    # In[ ]:


    r = 5
    roms_irka = []
    for mu in mu_list:
        irka_reductor = IRKAReductor(so_sys.to_lti(), mu=mu)
        rom_irka = irka_reductor.reduce(r)
        roms_irka.append(rom_irka)


    # In[ ]:


    fig, ax = plt.subplots()
    for rom_irka in roms_irka:
        poles_rom_irka = rom_irka.poles()
        ax.plot(poles_rom_irka.real, poles_rom_irka.imag, '.', label=fr'$\mu = {mu}$')
    ax.set_title("IRKA reduced model's poles")
    plt.show()


    # In[ ]:


    for mu, rom_irka in zip(mu_list, roms_irka):
        err_irka = so_sys - rom_irka
        print(f'mu = {mu}')
        print(f'    IRKA relative H_2-error:    {err_irka.h2_norm(mu=mu) / so_sys.h2_norm(mu=mu):e}')
        if config.HAVE_SLYCOT:
            print(f'    IRKA relative H_inf-error:  {err_irka.hinf_norm(mu=mu) / so_sys.hinf_norm(mu=mu):e}')
        print(f'    IRKA relative Hankel-error: {err_irka.hankel_norm(mu=mu) / so_sys.hankel_norm(mu=mu):e}')


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_irka in zip(mu_list, roms_irka):
        rom_irka.mag_plot(w, ax=ax, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of IRKA reduced models')
    ax.legend()
    plt.show()


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_irka in zip(mu_list, roms_irka):
        (so_sys - rom_irka).mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of the IRKA error system')
    ax.legend()
    plt.show()


    # ## Second-Order Iterative Rational Krylov Algorithm (SOR-IRKA)

    # In[ ]:


    r = 5
    roms_sor_irka = []
    for mu in mu_list:
        sor_irka_reductor = SORIRKAReductor(so_sys, mu=mu)
        rom_sor_irka = sor_irka_reductor.reduce(r)
        roms_sor_irka.append(rom_sor_irka)


    # In[ ]:


    fig, ax = plt.subplots()
    for rom_sor_irka in roms_sor_irka:
        poles_rom_sor_irka = rom_sor_irka.poles()
        ax.plot(poles_rom_sor_irka.real, poles_rom_sor_irka.imag, '.', label=fr'$\mu = {mu}$')
    ax.set_title("SORIRKA reduced model's poles")
    plt.show()


    # In[ ]:


    for mu, rom_sor_irka in zip(mu_list, roms_sor_irka):
        err_sor_irka = so_sys - rom_sor_irka
        print(f'mu = {mu}')
        print(f'    SORIRKA relative H_2-error:    {err_sor_irka.h2_norm(mu=mu) / so_sys.h2_norm(mu=mu):e}')
        if config.HAVE_SLYCOT:
            print(f'    SORIRKA relative H_inf-error:  {err_sor_irka.hinf_norm(mu=mu) / so_sys.hinf_norm(mu=mu):e}')
        print(f'    SORIRKA relative Hankel-error: {err_sor_irka.hankel_norm(mu=mu) / so_sys.hankel_norm(mu=mu):e}')


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_sor_irka in zip(mu_list, roms_sor_irka):
        rom_sor_irka.mag_plot(w, ax=ax, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of SORIRKA reduced models')
    ax.legend()
    plt.show()


    # In[ ]:


    fig, ax = plt.subplots()
    for mu, rom_sor_irka in zip(mu_list, roms_sor_irka):
        (so_sys - rom_sor_irka).mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
    ax.set_title('Magnitude plot of the SORIRKA error system')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    run_demo()
