# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import product

import pytest

from pymor.discretizers.builtin import discretize_stationary_cg, discretize_instationary_fv
from pymortests.fixtures.analyticalproblem import (picklable_thermalblock_problems, non_picklable_thermalblock_problems,
                                                   burgers_problems)


stationary_cg_generators = \
    [lambda p=p, d=d: discretize_stationary_cg(p, diameter=d)[0]
     for p, d in product(picklable_thermalblock_problems, [1./50., 1./100.])]

picklable_model_generators = stationary_cg_generators \
    + [lambda p=p, d=d: discretize_instationary_fv(p, diameter=d, nt=100)[0]
       for p, d in product(burgers_problems, [1./10., 1./15.])]


non_picklable_model_generators = \
    [lambda p=p, d=d: discretize_stationary_cg(p, diameter=d)[0]
     for p, d in product(non_picklable_thermalblock_problems, [1./20., 1./30.])]


model_generators = picklable_model_generators + non_picklable_model_generators


@pytest.fixture(params=model_generators)
def model(request):
    return request.param()


@pytest.fixture(params=[lambda p=p, d=d: discretize_stationary_cg(p, diameter=d)[0]
                        for p, d in product(non_picklable_thermalblock_problems, [1./20., 1./30.])])
def stationary_models(request):
    return request.param()


@pytest.fixture(params=picklable_model_generators)
def picklable_model(request):
    return request.param()
