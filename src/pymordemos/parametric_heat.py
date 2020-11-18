#!/usr/bin/env python
# coding: utf-8
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# In[ ]:




# In[ ]:


import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
import matplotlib as mpl

from pymor.basic import *
from pymor.core.config import config

from pymor.core.logger import set_log_levels
set_log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING'})

set_defaults({'pymor.discretizers.builtin.gui.jupyter.get_visualizer.backend': 'not pythreejs'})


# # Model

# In[ ]:


p = InstationaryProblem(
    StationaryProblem(
        domain=LineDomain([0.,1.], left='robin', right='robin'),
        diffusion=LincombFunction([ExpressionFunction('(x[...,0] <= 0.5) * 1.', 1),
                                   ExpressionFunction('(0.5 < x[...,0]) * 1.', 1)],
                                  [1,
                                   ProjectionParameterFunctional('diffusion')]),
        robin_data=(ConstantFunction(1., 1), ExpressionFunction('(x[...,0] < 1e-10) * 1.', 1)),
        outputs=(('l2_boundary', ExpressionFunction('(x[...,0] > (1 - 1e-10)) * 1.', 1)),),
    ),
    ConstantFunction(0., 1),
    T=3.
)

fom, _ = discretize_instationary_cg(p, diameter=1/100, nt=100)


# In[ ]:


fom.visualize(fom.solve(mu=0.1))


# In[ ]:


fom.visualize(fom.solve(mu=1))


# In[ ]:


fom.visualize(fom.solve(mu=10))


# In[ ]:


lti = fom.to_lti()


# # System analysis

# In[ ]:


print(f'order of the model = {lti.order}')
print(f'number of inputs   = {lti.dim_input}')
print(f'number of outputs  = {lti.dim_output}')


# In[ ]:


mu_list = [0.1, 1, 10]

fig, ax = plt.subplots(len(mu_list), 1, sharex=True, sharey=True)
for i, mu in enumerate(mu_list):
    poles = lti.poles(mu=mu)
    ax[i].plot(poles.real, poles.imag, '.')
    ax[i].set_xscale('symlog')
    ax[i].set_title(fr'$\mu = {mu}$')
fig.suptitle('System poles')
fig.subplots_adjust(hspace=0.5)
plt.show()


# In[ ]:


mu_list = [0.1, 1, 10]

fig, ax = plt.subplots()
w = np.logspace(-1, 3, 100)
for mu in mu_list:
    lti.mag_plot(w, ax=ax, mu=mu, label=fr'$\mu = {mu}$')
ax.legend()
plt.show()


# In[ ]:


w_list = np.logspace(-1, 3, 100)
mu_list = np.logspace(-1, 1, 20)

lti_w_mu = np.zeros((len(w_list), len(mu_list)))
for i, mu in enumerate(mu_list):
    lti_w_mu[:, i] = spla.norm(lti.freq_resp(w, mu=mu), axis=(1, 2))


# In[ ]:


fig, ax = plt.subplots()
out = ax.contourf(w_list, mu_list, lti_w_mu.T,
                  norm=mpl.colors.LogNorm(),
                  levels=np.logspace(-16, np.log10(lti_w_mu.max()), 100))
ax.set_xlabel(r'Frequency $\omega$')
ax.set_ylabel(r'Parameter $\mu$')
ax.set_xscale('log')
ax.set_yscale('log')
fig.colorbar(out, ticks=np.logspace(-16, 0, 17))
plt.show()


# In[ ]:


mu_list = [0.1, 1, 10]

fig, ax = plt.subplots()
for mu in mu_list:
    hsv = lti.hsv(mu=mu)
    ax.semilogy(range(1, len(hsv) + 1), hsv, label=fr'$\mu = {mu}$')
ax.set_title('Hankel singular values')
ax.legend()
plt.show()


# In[ ]:


fig, ax = plt.subplots()
mu_fine = np.logspace(-1, 1, 20)
h2_norm_mu = [lti.h2_norm(mu=mu) for mu in mu_fine]
ax.plot(mu_fine, h2_norm_mu, label=r'$\mathcal{H}_2$-norm')

if config.HAVE_SLYCOT:
    hinf_norm_mu = [lti.hinf_norm(mu=mu) for mu in mu_fine]
    ax.plot(mu_fine, hinf_norm_mu, label=r'$\mathcal{H}_\infty$-norm')

hankel_norm_mu = [lti.hankel_norm(mu=mu) for mu in mu_fine]
ax.plot(mu_fine, hankel_norm_mu, label='Hankel norm')

ax.set_xlabel(r'$\mu$')
ax.set_title('System norms')
ax.legend()
plt.show()


# # Balanced truncation

# In[ ]:


def reduction_errors(lti, r, mu_fine, reductor, **kwargs):
    h2_err_mu = []
    hinf_err_mu = []
    hankel_err_mu = []
    for mu in mu_fine:
        rom_mu = reductor(lti, mu=mu, **kwargs).reduce(r)
        h2_err_mu.append((lti - rom_mu).h2_norm(mu=mu) / lti.h2_norm(mu=mu))
        if config.HAVE_SLYCOT:
            hinf_err_mu.append((lti - rom_mu).hinf_norm(mu=mu) / lti.hinf_norm(mu=mu))
        hankel_err_mu.append((lti - rom_mu).hankel_norm(mu=mu) / lti.hankel_norm(mu=mu))
    return h2_err_mu, hinf_err_mu, hankel_err_mu


# In[ ]:


r = 5
mu_fine = np.logspace(-1, 1, 10)
h2_bt_err_mu, hinf_bt_err_mu, hankel_bt_err_mu = reduction_errors(lti, r, mu_fine, BTReductor)


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


h2_irka_err_mu, hinf_irka_err_mu, hankel_irka_err_mu = reduction_errors(lti, r, mu_fine, IRKAReductor)


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


# # Two-Sided Iteration Algorithm (TSIA)

# In[ ]:


h2_tsia_err_mu, hinf_tsia_err_mu, hankel_tsia_err_mu = reduction_errors(lti, r, mu_fine, TSIAReductor)


# In[ ]:


fig, ax = plt.subplots()
ax.semilogy(mu_fine, h2_tsia_err_mu, '.-', label=r'$\mathcal{H}_2$')
if config.HAVE_SLYCOT:
    ax.semilogy(mu_fine, hinf_tsia_err_mu, '.-', label=r'$\mathcal{H}_\infty$')
ax.semilogy(mu_fine, hankel_tsia_err_mu, '.-', label='Hankel')

ax.set_xlabel(r'$\mu$')
ax.set_title('TSIA errors')
ax.legend()
plt.show()


# # One-sided IRKA

# In[ ]:


h2_osirka_err_mu, hinf_osirka_err_mu, hankel_osirka_err_mu = reduction_errors(
    lti, r, mu_fine, OneSidedIRKAReductor, version='V'
)


# In[ ]:


fig, ax = plt.subplots()
ax.semilogy(mu_fine, h2_osirka_err_mu, '.-', label=r'$\mathcal{H}_2$')
if config.HAVE_SLYCOT:
    ax.semilogy(mu_fine, hinf_osirka_err_mu, '.-', label=r'$\mathcal{H}_\infty$')
ax.semilogy(mu_fine, hankel_osirka_err_mu, '.-', label='Hankel')

ax.set_xlabel(r'$\mu$')
ax.set_title('One-sided IRKA errors')
ax.legend()
plt.show()
