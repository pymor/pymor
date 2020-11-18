#!/usr/bin/env python
# coding: utf-8
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# In[ ]:


import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps
import matplotlib.pyplot as plt
import matplotlib as mpl

from pymor.core.config import config
from pymor.models.iosys import LTIModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.bt import BTReductor
from pymor.reductors.h2 import IRKAReductor


# # Model
#
# https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Synthetic_parametric_model

# In[ ]:


n = 100  # order of the resulting system

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


# In[ ]:


A0 = NumpyMatrixOperator(A0)
Amu = NumpyMatrixOperator(Amu)
B = NumpyMatrixOperator(B)
C = NumpyMatrixOperator(C)


# In[ ]:


A = A0 + Amu * ProjectionParameterFunctional('mu')


# In[ ]:


lti = LTIModel(A, B, C)


# # Magnitude plot

# In[ ]:


mu_list_short = [1/50, 1/20, 1/10, 1/5, 1/2, 1]


# In[ ]:


w = np.logspace(0.5, 3.5, 200)

fig, ax = plt.subplots()
for mu in mu_list_short:
    lti.mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
ax.legend()
plt.show()


# In[ ]:


w_list = np.logspace(0.5, 3.5, 200)
mu_list = np.linspace(1/50, 1, 50)

lti_w_mu = np.zeros((len(w_list), len(mu_list)))
for i, mu in enumerate(mu_list):
    lti_w_mu[:, i] = spla.norm(lti.freq_resp(w_list, mu=mu), axis=(1, 2))


# In[ ]:


fig, ax = plt.subplots()
out = ax.contourf(w_list, mu_list, lti_w_mu.T,
                  norm=mpl.colors.LogNorm(),
                  levels=np.logspace(np.log10(lti_w_mu.min()), np.log10(lti_w_mu.max()), 100))
ax.set_xlabel(r'Frequency $\omega$')
ax.set_ylabel(r'Parameter $\mu$')
ax.set_xscale('log')
#ax.set_yscale('log')
fig.colorbar(out, ticks=np.logspace(-2, 1, 7))
plt.show()


# # Hankel singular values

# In[ ]:


fig, ax = plt.subplots()
for mu in mu_list_short:
    hsv = lti.hsv(mu=mu)
    ax.semilogy(range(1, len(hsv) + 1), hsv, '.-', label=fr'$\mu = {mu}$')
ax.set_title('Hankel singular values')
ax.legend()
plt.show()


# # System norms

# In[ ]:


fig, ax = plt.subplots()
mu_fine = np.linspace(1/50, 1, 20)
h2_norm_mu = [lti.h2_norm(mu=mu) for mu in mu_fine]
ax.plot(mu_fine, h2_norm_mu, '.-', label=r'$\mathcal{H}_2$-norm')

if config.HAVE_SLYCOT:
    hinf_norm_mu = [lti.hinf_norm(mu=mu) for mu in mu_fine]
    ax.plot(mu_fine, hinf_norm_mu, '.-', label=r'$\mathcal{H}_\infty$-norm')

hankel_norm_mu = [lti.hankel_norm(mu=mu) for mu in mu_fine]
ax.plot(mu_fine, hankel_norm_mu, '.-', label='Hankel norm')

ax.set_xlabel(r'$\mu$')
ax.set_title('System norms')
ax.legend()
plt.show()


# # Balanced truncation

# In[ ]:


def reduction_errors(lti, r, mu_fine, method):
    h2_err_mu = []
    hinf_err_mu = []
    hankel_err_mu = []
    for mu in mu_fine:
        rom_mu = method(lti, r, mu=mu)
        h2_err_mu.append((lti - rom_mu).h2_norm(mu=mu) / lti.h2_norm(mu=mu))
        if config.HAVE_SLYCOT:
            hinf_err_mu.append((lti - rom_mu).hinf_norm(mu=mu) / lti.hinf_norm(mu=mu))
        hankel_err_mu.append((lti - rom_mu).hankel_norm(mu=mu) / lti.hankel_norm(mu=mu))
    return h2_err_mu, hinf_err_mu, hankel_err_mu


# In[ ]:


r = 20
mu_fine = np.linspace(1/50, 1, 10)
(
    h2_bt_err_mu,
    hinf_bt_err_mu,
    hankel_bt_err_mu
) = reduction_errors(lti, r, mu_fine,
                     lambda lti, r, mu=None: BTReductor(lti, mu=mu).reduce(r))


# In[ ]:


fig, ax = plt.subplots()
ax.semilogy(mu_fine, h2_bt_err_mu, '.-', label=r'$\mathcal{H}_2$')
if config.HAVE_SLYCOT:
    ax.semilogy(mu_fine, hinf_bt_err_mu, '.-', label=r'$\mathcal{H}_\infty$')
ax.semilogy(mu_fine, hankel_bt_err_mu, '.-', label='Hankel')

ax.set_xlabel(r'$\mu$')
ax.set_title('Balanced truncation errors')
ax.legend()
plt.show()


# # Iterative Rational Krylov Algorithm (IRKA)

# In[ ]:


(
    h2_irka_err_mu,
    hinf_irka_err_mu,
    hankel_irka_err_mu
) = reduction_errors(lti, r, mu_fine,
                     lambda lti, r, mu=mu: IRKAReductor(lti, mu=mu).reduce(r, conv_crit='h2'))


# In[ ]:


fig, ax = plt.subplots()
ax.semilogy(mu_fine, h2_irka_err_mu, '.-', label=r'$\mathcal{H}_2$')
if config.HAVE_SLYCOT:
    ax.semilogy(mu_fine, hinf_irka_err_mu, '.-', label=r'$\mathcal{H}_\infty$')
ax.semilogy(mu_fine, hankel_irka_err_mu, '.-', label='Hankel')

ax.set_xlabel(r'$\mu$')
ax.set_title('IRKA errors')
ax.legend()
plt.show()
