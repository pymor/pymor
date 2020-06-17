# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Example script for the usage of neural networks in model order reduction (approach by Hesthaven and Ubbiali)

Usage:
    neural_networks.py [--vis] DIM N ORDER TRAINING_SAMPLES VALIDATION_SAMPLES

Arguments:
    DIM                  Spatial dimension of the problem.
    N                    Number of mesh intervals per spatial dimension.
    ORDER                Finite element order.
    TRAINING_SAMPLES     Number of samples used for training the neural network.
    VALIDATION_SAMPLES   Number of samples used for validation during the training phase.

Options:
    -h, --help   Show this message.
    --vis        Visualize full order solution and reduced solution for a test set.
"""

from docopt import docopt

import numpy as np

from pymor.basic import *

from pymor.core.config import config


def create_fom(DIM, N, ORDER):
    import dolfin as df

    if DIM == 2:
        mesh = df.UnitSquareMesh(N, N)
    elif DIM == 3:
        mesh = df.UnitCubeMesh(N, N, N)
    else:
        raise NotImplementedError

    V = df.FunctionSpace(mesh, "CG", ORDER)

    g = df.Constant(1.0)
    c = df.Constant(1.)

    class DirichletBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - 1.0) < df.DOLFIN_EPS and on_boundary
    db = DirichletBoundary()
    bc = df.DirichletBC(V, g, db)

    u = df.Function(V)
    v = df.TestFunction(V)
    f = df.Expression("x[0]*sin(x[1])", degree=2)
    F = df.inner((1 + c*u**2)*df.grad(u), df.grad(v))*df.dx - f*v*df.dx

    df.solve(F == 0, u, bc,
             solver_parameters={"newton_solver": {"relative_tolerance": 1e-6}})

    from pymor.bindings.fenics import FenicsVectorSpace, FenicsOperator, FenicsVisualizer

    space = FenicsVectorSpace(V)
    op = FenicsOperator(F, space, space, u, (bc,),
                        parameter_setter=lambda mu: c.assign(mu['c'].item()),
                        parameters={'c': 1},
                        solver_options={'inverse': {'type': 'newton', 'rtol': 1e-6}})
    rhs = VectorOperator(op.range.zeros())

    fom = StationaryModel(op, rhs,
                          visualizer=FenicsVisualizer(space))

    return fom


def neural_networks_demo(args):
    logger = getLogger('pymordemos.neural_networks')

    if not config.HAVE_TORCH:
        logger.error('PyTorch is not installed! Stopping.')
        return

    TRAINING_SAMPLES = args['TRAINING_SAMPLES']
    VALIDATION_SAMPLES = args['VALIDATION_SAMPLES']

    DIM = int(args['DIM'])
    N = int(args['N'])
    ORDER = int(args['ORDER'])

    fom = create_fom(DIM, N, ORDER)

    parameter_space = fom.parameters.space((0, 1000.))

    from pymor.reductors.neural_network import NeuralNetworkReductor

    training_set = parameter_space.sample_uniformly(int(TRAINING_SAMPLES))
    validation_set = parameter_space.sample_randomly(int(VALIDATION_SAMPLES))

    basis_size = 10

    reductor = NeuralNetworkReductor(fom,
                                     training_set,
                                     validation_set,
                                     basis_size=basis_size,
                                     hidden_layers='[(N+P)*3, (N+P)*3, (N+P)*3]',
                                     restarts=100)
    rom = reductor.reduce()

    test_set = parameter_space.sample_randomly(10)

    speedups = []

    import time

    print(f'Performing test on set of size {len(test_set)} ...')

    U = fom.solution_space.empty(reserve=len(test_set))
    U_red = fom.solution_space.empty(reserve=len(test_set))

    for mu in test_set:
        tic = time.time()
        U.append(fom.solve(mu))
        time_fom = time.time() - tic

        tic = time.time()
        U_red.append(reductor.reconstruct(rom.solve(mu)))
        time_red = time.time() - tic

        speedups.append(time_fom / time_red)

    absolute_errors = (U - U_red).l2_norm()
    relative_errors = (U - U_red).l2_norm() / U.l2_norm()

    if args['--vis']:
        fom.visualize((U, U_red),
                      legend=('Full solution', 'Reduced solution'))

    print(f'Average absolute error: {np.average(absolute_errors)}')
    print(f'Average relative error: {np.average(relative_errors)}')
    print(f'Median of speedup: {np.median(speedups)}')


if __name__ == '__main__':
    args = docopt(__doc__)
    neural_networks_demo(args)
