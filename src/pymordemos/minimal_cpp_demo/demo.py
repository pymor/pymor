# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.pod import pod
from pymor.algorithms.timestepping import ExplicitEulerTimeStepper
from pymor.discretizers.builtin.gui.visualizers import OnedVisualizer
from pymor.models.basic import InstationaryModel
from pymor.discretizers.builtin import OnedGrid
from pymor.operators.constructions import VectorOperator, LincombOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.basic import InstationaryRBReductor

# import wrapped classes
from pymordemos.minimal_cpp_demo.wrapper import WrappedDiffusionOperator


def discretize(n, nt, blocks):
    h = 1. / blocks
    ops = [WrappedDiffusionOperator.create(n, h * i, h * (i + 1)) for i in range(blocks)]
    pfs = [ProjectionParameterFunctional('diffusion_coefficients', blocks, i) for i in range(blocks)]
    operator = LincombOperator(ops, pfs)

    initial_data = operator.source.zeros()

    # use data property of WrappedVector to setup rhs
    # note that we cannot use the data property of ListVectorArray,
    # since ListVectorArray will always return a copy
    rhs_vec = operator.range.zeros()
    rhs_data = rhs_vec.vectors[0].to_numpy()
    rhs_data[:] = np.ones(len(rhs_data))
    rhs_data[0] = 0
    rhs_data[len(rhs_data) - 1] = 0
    rhs = VectorOperator(rhs_vec)

    # hack together a visualizer ...
    grid = OnedGrid(domain=(0, 1), num_intervals=n)
    visualizer = OnedVisualizer(grid)

    time_stepper = ExplicitEulerTimeStepper(nt)

    fom = InstationaryModel(T=1e-0, operator=operator, rhs=rhs, initial_data=initial_data,
                            time_stepper=time_stepper, num_values=20,
                            visualizer=visualizer, name='C++-Model')
    return fom


# discretize
fom = discretize(50, 10000, 4)
parameter_space = fom.parameters.space(0.1, 1)

# generate solution snapshots
snapshots = fom.solution_space.empty()
for mu in parameter_space.sample_uniformly(2):
    snapshots.append(fom.solve(mu))

# apply POD
reduced_basis = pod(snapshots, modes=4)[0]

# reduce the model
reductor = InstationaryRBReductor(fom, reduced_basis, check_orthonormality=True)
rom = reductor.reduce()

# stochastic error estimation
mu_max = None
err_max = -1.
for mu in parameter_space.sample_randomly(10):
    U_RB = (reductor.reconstruct(rom.solve(mu)))
    U = fom.solve(mu)
    err = np.max((U_RB-U).norm())
    if err > err_max:
        err_max = err
        mu_max = mu

# visualize maximum error solution
U_RB = (reductor.reconstruct(rom.solve(mu_max)))
U = fom.solve(mu_max)

# avoid blocking in CI run
if __name__ == '__main__':
    fom.visualize((U_RB, U), title=f'mu = {mu}', legend=('reduced', 'detailed'))
    fom.visualize((U-U_RB), title=f'mu = {mu}', legend=('error'))
