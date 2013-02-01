from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import pymor.core as core
from pymor.domaindescriptions import BoundaryType
from pymor.domaindescriptions import RectDomain
from pymor.functions import ConstantFunction


class PoissonProblem(object):

    def __init__(self, domain=RectDomain(), rhs=ConstantFunction(dim_domain=2),
                 diffusion_functions=(ConstantFunction(dim_domain=2),),
                 diffusion_functionals=None,
                 dirichlet_data=ConstantFunction(value=0, dim_domain=2), parameter_dependent=None):
        self.domain = domain
        self.rhs = rhs
        self.diffusion_functions = diffusion_functions
        self.diffusion_functionals = diffusion_functionals
        self.dirichlet_data = dirichlet_data
        if parameter_dependent is None:
            if diffusion_functionals is not None:
                self.parameter_dependent = True
            else:
                self.parameter_dependent = len(diffusion_functions) > 1
        else:
            self.parameter_dependent = parameter_independent
