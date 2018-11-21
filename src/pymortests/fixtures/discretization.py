# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from itertools import product

import pytest
from pkg_resources import resource_filename

from pymor.discretizers.cg import discretize_stationary_cg
from pymor.discretizers.fv import discretize_instationary_fv
from pymor.discretizers.disk import discretize_stationary_from_disk, discretize_instationary_from_disk
from pymortests.fixtures.analyticalproblem import (picklable_thermalblock_problems, non_picklable_thermalblock_problems,
                                                   burgers_problems)


picklable_discretizaion_generators = \
        [lambda p=p, d=d: discretize_stationary_cg(p, diameter=d)[0]
         for p, d in product(picklable_thermalblock_problems, [1./50., 1./100.])] + \
        [lambda p=p, d=d: discretize_instationary_fv(p, diameter=d, nt=100)[0]
         for p, d in product(burgers_problems, [1./10., 1./15.])] + \
        [lambda p=p: discretize_stationary_from_disk(parameter_file=p)
         for p in (resource_filename('pymortests', 'testdata/parameter_stationary.ini'),)] + \
        [lambda p=p: discretize_instationary_from_disk(parameter_file=p)
         for p in (resource_filename('pymortests', 'testdata/parameter_instationary.ini'),)]


non_picklable_discretization_generators = \
        [lambda p=p, d=d: discretize_stationary_cg(p, diameter=d)[0]
         for p, d in product(non_picklable_thermalblock_problems, [1./20., 1./30.])]


discretization_generators = picklable_discretizaion_generators + non_picklable_discretization_generators


@pytest.fixture(params=discretization_generators)
def discretization(request):
    return request.param()


@pytest.fixture(params=picklable_discretizaion_generators)
def picklable_discretization(request):
    return request.param()
