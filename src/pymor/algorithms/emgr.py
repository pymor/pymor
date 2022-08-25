# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.pod import pod
from pymor.analyticalproblems.functions import ExpressionFunction
from pymor.operators.interface import Operator
from pymor.operators.constructions import LowRankOperator


def emgr(model, *args, **kwargs):

    dim_input = model.dim_input
    Xm = model.solution_space.empty()
    for m in range(dim_input):
        exprs = ['(x[0] <= 0.1)*1.' if mm == m else '0'
                 for mm in range(dim_input)]
        expr = '[' + ','.join(exprs) + ']'
        input_function = ExpressionFunction(expr, 1)
        xm = model.solve(input=input_function)
        Xm.append(xm, remove_from_other=True)

    modes, svals = pod(Xm, rtol=1e-5)

    lr_ops = LowRankOperator(modes, np.diag(svals**2), modes)

    return lr_ops
