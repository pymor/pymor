# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.basic import *
from pymor.algorithms.emgr import emgr
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
import numpy as np


A = NumpyMatrixOperator(np.eye(5) * 1)
B = VectorOperator(A.source.from_numpy(np.ones(5))) * ProjectionParameterFunctional('input', 2, 0)

model = InstationaryModel(1, A.source.zeros(), A, B, time_stepper=ImplicitEulerTimeStepper(10))

Wc = emgr(model)
