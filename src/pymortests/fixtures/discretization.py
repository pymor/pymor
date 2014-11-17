# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function


from itertools import product

import pytest

from pymor.discretizers.advection import discretize_nonlinear_instationary_advection_fv
from pymor.discretizers.elliptic import discretize_elliptic_cg
from pymortests.fixtures.analyticalproblem import (picklable_thermalblock_problems, non_picklable_thermalblock_problems,
                                                   burgers_problems)


picklable_discretizaion_generators = \
        [lambda p=p,d=d: discretize_elliptic_cg(p, diameter=d)[0]
         for p, d in product(picklable_thermalblock_problems, [1./50., 1./100.])] + \
        [lambda p=p,d=d: discretize_nonlinear_instationary_advection_fv(p, diameter=d)[0]
         for p, d in product(burgers_problems, [1./10., 1./15.])]


non_picklable_discretization_generators = \
        [lambda p=p,d=d: discretize_elliptic_cg(p, diameter=d)[0]
         for p, d in product(non_picklable_thermalblock_problems, [1./20., 1./30.])]


discretization_generators = picklable_discretizaion_generators + non_picklable_discretization_generators


@pytest.fixture(params=discretization_generators)
def discretization(request):
    return request.param()


@pytest.fixture(params=picklable_discretizaion_generators)
def picklable_discretization(request):
    return request.param()
