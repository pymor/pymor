from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import pymor.core as core
from pymor.common import BoundaryType
from pymor.common.domaindescription import Rect
from pymor.common.function.nonparametric import Constant as ConstantFunc


class Poisson(object):

    def __init__(self, domain=Rect(), rhs=ConstantFunc(dim_domain=2), diffusion_functions=(ConstantFunc(dim_domain=2),),
                 dirichlet_data=ConstantFunc(value=0, dim_domain=2), parameter_dependent=None):
        self.domain = domain
        self.rhs = rhs
        self.diffusion_functions = diffusion_functions
        self.dirichlet_data = dirichlet_data
        if parameter_dependent is None:
            self.parameter_dependent = len(diffusion_functions) > 1
        else:
            self.parameter_dependent = parameter_independent
