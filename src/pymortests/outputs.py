# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from pymor.basic import *

def test_output_estimate():
    grid_intervals = 10
    training_samples = 10
    random_samples = 10

    # a real valued output from the optimization demo
    fom_1 = create_fom(grid_intervals, vector_valued_output=False, parametric_objective=True)

    # a vector valued output (with BlockColumnOperator)
    fom_2 = create_fom(grid_intervals, vector_valued_output=True, parametric_objective=False)

    # an output which is actually a lincomb operator
    dim_source = fom_1.output_functional.source.dim
    random_matrix_1 = np.random.rand(2, dim_source)
    random_matrix_2 = np.random.rand(2, dim_source)
    op1 = NumpyMatrixOperator(random_matrix_1, source_id='STATE')
    op2 = NumpyMatrixOperator(random_matrix_2, source_id='STATE')
    ops = [op1, op2]
    lincomb_op = LincombOperator(ops, [1., fom_1.output_functional.coefficients[0]])
    fom_3 = fom_2.with_(output_functional=lincomb_op)

    # all foms with different output_functionals
    foms = [fom_1, fom_2, fom_3]

    # only use CoerciveRBReductor for now
    reductor = CoerciveRBReductor

    # Parameter space and operator are equal for all foms
    parameter_space = fom_1.parameters.space(0.1, 1)
    training_set = parameter_space.sample_randomly(training_samples)
    random_set = parameter_space.sample_randomly(random_samples, seed=11)
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', fom_1.parameters)

    estimator_values = []
    for fom in foms:
        print('_____NEW FOM ____')
        for operator_symmetric in [True, False]:
            for construct_dual_bases in [False, True]:
                print(f'operator_symmetric: {operator_symmetric}, construct_dual_bases {construct_dual_bases}')
                # generate solution snapshots
                primal_snapshots = fom.solution_space.empty()
                dual_snapshotss = [fom.solution_space.empty() for d in range(fom.dim_output)]

                # construct training data
                for mu in training_set:
                    primal_snapshots.append(fom.solve(mu))

                for d in range(fom.dim_output):
                    # initialize dual model from reductor
                    dual_fom = reductor.dual_model(fom, d, operator_symmetric)
                    for mu in training_set:
                        dual_snapshotss[d].append(dual_fom.solve(mu))

                # apply POD on bases
                primal_reduced_basis = pod(primal_snapshots, modes=6)[0]
                if construct_dual_bases:
                    dual_reduced_bases = []
                    for d in range(fom.dim_output):
                        dual_reduced_bases.append(pod(dual_snapshotss[d], modes=6)[0])
                else:
                    dual_reduced_bases = None


                RB_reductor = reductor(fom,
                                       RB=primal_reduced_basis,
                                       coercivity_estimator=coercivity_estimator,
                                       operator_is_symmetric=operator_symmetric,
                                       dual_bases=dual_reduced_bases)

                # two different roms
                rom_standard = RB_reductor.reduce()
                rom_restricted = RB_reductor.reduce(2)

                for mu in random_set:
                    s_fom = fom.output(mu=mu)
                    for rom in [rom_standard, rom_restricted]:
                        s_rom, s_est = rom.output(return_error_estimate=True, mu=mu)
                        estimator_values.append(s_est)
                        for s_r, s_f, s_e in np.dstack((s_rom, s_fom, s_est))[0]:
                            # print(rom.solution_space.dim, np.abs(s_r-s_f), s_e)
                            assert np.abs(s_r-s_f) <= s_e + 1e-12

def create_fom(grid_intervals, vector_valued_output=False, parametric_objective=False):
    p = thermal_block_problem([2,2])
    f = ConstantFunction(1, dim_domain=2)

    if parametric_objective:
        theta_J = ExpressionParameterFunctional('1 + 1/5 * diffusion[0] + 1/5 * diffusion[1]', p.parameters)
    else:
        theta_J = 1.

    if vector_valued_output:
        p = p.with_(outputs=[('l2', f * theta_J), ('l2', f * 0.5 * theta_J)])
    else:
        p = p.with_(outputs=[('l2', f * theta_J)])

    fom, _ = discretize_stationary_cg(p, diameter=1./grid_intervals)
    return fom

