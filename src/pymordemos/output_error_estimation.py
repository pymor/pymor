#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import numpy as np
from typer import Argument, run

from pymor.basic import *
from pymordemos.linear_optimization import create_fom

def main(
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    training_samples: int = Argument(..., help='Number of samples used for training the reduced basis.')
    verification_samples: int = Argument(..., help='Number of samples used for verification of the output error.')
):
    """Example script for using the DWR output error estimation"""
    # a real valued output from the optimization demo
    fom_1, mu_bar = create_fom(grid_intervals, vector_valued_output=False)

    # a non parametric output functional
    fom_2 = fom_1.with_(output_functional=fom_1.output_functional.with_(coefficients=[1.]))

    # a vector valued output (with BlockColumnOperator)
    fom_3, _ = create_fom(grid_intervals, vector_valued_output=True)

    # an output which is actually a lincomb operator
    dim_source = fom_1.output_functional.source.dim
    random_matrix_1 = np.random.rand(2, dim_source)
    random_matrix_2 = np.random.rand(2, dim_source)
    op1 = NumpyMatrixOperator(random_matrix_1, source_id='STATE')
    op2 = NumpyMatrixOperator(random_matrix_2, source_id='STATE')
    ops = [op1, op2]
    lincomb_op = LincombOperator(ops, [1., fom_1.output_functional.coefficients[0]])
    fom_4 = fom_3.with_(output_functional=lincomb_op)

    # all foms with different output_functionals
    foms = [fom_1, fom_2, fom_3, fom_4]

    # only use CoerciveRBReductor for now
    reductor = CoerciveRBReductor

    # Parameter space and operator are equal for all foms
    parameter_space = fom_1.parameters.space(0, np.pi)

    training_set = parameter_space.sample_uniformly(training_samples)
    random_set = parameter_space.sample_randomly(verification_samples, seed=0)
    coercivity_estimator = MinThetaParameterFunctional(fom_1.operator.coefficients, mu_bar)

    estimator_values = []
    for fom in foms:
        for dual in [True, False]:
            RB_reductor = reductor(fom, product=fom.energy_product,
                                   coercivity_estimator=coercivity_estimator,
                                   operator_is_symmetric=dual)
            RB_greedy_data = rb_greedy(fom, RB_reductor, training_set, atol=1e-2)

            # two different roms
            rom_standard = RB_greedy_data['rom']
            rom_restricted = RB_reductor.reduce(2)

            for mu in random_set:
                s_fom = fom.output(mu=mu)
                for rom in [rom_standard]: #, rom_restricted]:
                    s_rom, s_est = rom.output(return_error_estimate=True, mu=mu)
                    estimator_values.append(s_est)
                    for s_r, s_f, s_e in np.dstack((s_rom, s_fom, s_est))[0]:
                        print(np.abs(s_r-s_f), s_e)
                        assert np.abs(s_r-s_f) <= s_e


if __name__ == '__main__':
    run(main)
