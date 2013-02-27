from __future__ import absolute_import, division, print_function

import numpy as np

import pymor.core as core
from pymor.tools import Named
from pymor.domaindescriptions import BoundaryType
from pymor.domaindescriptions import RectDomain
from pymor.functions import ConstantFunction


class EllipticProblem(core.BasicInterface, Named):
    r'''Standard elliptic analytical problem.

    The problem consists in solving

    .. :math  - \nabla \cdot (\sum_{k=0}^K \theta_k(\mu) * d_k(x) \nabla u(x, \mu)) = f(x, \mu)

    for u.

    Parameters
    ----------
    domain
        A domain description of the domain the problem is posed on.
    rhs
        The function f(x, mu).
    diffusion_functions
        List of the functions d_k(x).
    diffusion_functionals
        List of the functionals theta_k(mu). If None, and `len(diffusion_functions) > 1`
        let theta_k be the kth projection of the coefficient part of mu.
        If None and `len(diffusion_functions) == 1`, no parameter dependence is
        assumed.
    dirichlet_data
        Function providing the Dirichlet boundary values in global coordinates.
    name
        Name of the problem.
    '''

    def __init__(self, domain=RectDomain(), rhs=ConstantFunction(dim_domain=2),
                 diffusion_functions=(ConstantFunction(dim_domain=2),),
                 diffusion_functionals=None,
                 dirichlet_data=ConstantFunction(value=0, dim_domain=2), name=None):
        self.domain = domain
        self.rhs = rhs
        self.diffusion_functions = diffusion_functions
        self.diffusion_functionals = diffusion_functionals
        self.dirichlet_data = dirichlet_data
        self.name = name
