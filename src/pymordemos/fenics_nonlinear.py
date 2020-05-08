#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simple example script for reducing a FEniCS-based nonlinear diffusion problem.

Usage:
    fenics_nonlinear.py DIM N ORDER

Arguments:
    DIM               Spatial dimension of the problem.
    N                 Number of mesh intervals per spatial dimension.
    ORDER             Finite element order.

Options:
    -h, --help   Show this message.
"""

from docopt import docopt


def discretize(DIM, N, ORDER):
    # ### problem definition
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

    # ### pyMOR wrapping
    from pymor.bindings.fenics import FenicsVectorSpace, FenicsOperator, FenicsVisualizer
    from pymor.models.basic import StationaryModel
    from pymor.operators.constructions import VectorOperator

    space = FenicsVectorSpace(V)
    op = FenicsOperator(F, space, space, u, (bc,),
                        parameter_setter=lambda mu: c.assign(mu['c'].item()),
                        parameters={'c': 1},
                        solver_options={'inverse': {'type': 'newton', 'rtol': 1e-6}})
    rhs = VectorOperator(op.range.zeros())

    fom = StationaryModel(op, rhs,
                          visualizer=FenicsVisualizer(space))

    return fom


def fenics_nonlinear_demo(args):
    DIM = int(args['DIM'])
    N = int(args['N'])
    ORDER = int(args['ORDER'])

    from pymor.tools import mpi

    if mpi.parallel:
        from pymor.models.mpi import mpi_wrap_model
        local_models = mpi.call(mpi.function_call_manage, discretize, DIM, N, ORDER)
        fom = mpi_wrap_model(local_models, use_with=True, pickle_local_spaces=False)
    else:
        fom = discretize(DIM, N, ORDER)

    parameter_space = fom.parameters.space((0, 1000.))

    # ### ROM generation (POD/DEIM)
    from pymor.algorithms.ei import ei_greedy
    from pymor.algorithms.newton import newton
    from pymor.algorithms.pod import pod
    from pymor.operators.ei import EmpiricalInterpolatedOperator
    from pymor.reductors.basic import StationaryRBReductor

    U = fom.solution_space.empty()
    residuals = fom.solution_space.empty()
    for mu in parameter_space.sample_uniformly(10):
        UU, data = newton(fom.operator, fom.rhs.as_vector(), mu=mu, rtol=1e-6, return_residuals=True)
        U.append(UU)
        residuals.append(data['residuals'])

    dofs, cb, _ = ei_greedy(residuals, rtol=1e-7)
    ei_op = EmpiricalInterpolatedOperator(fom.operator, collateral_basis=cb, interpolation_dofs=dofs, triangular=True)

    rb, svals = pod(U, rtol=1e-7)
    fom_ei = fom.with_(operator=ei_op)
    reductor = StationaryRBReductor(fom_ei, rb)
    rom = reductor.reduce()
    # the reductor currently removes all solver_options so we need to add them again
    rom = rom.with_(operator=rom.operator.with_(solver_options=fom.operator.solver_options))

    # ### ROM validation
    import time
    import numpy as np

    # ensure that FFC is not called during runtime measurements
    rom.solve(1)

    errs = []
    speedups = []
    for mu in parameter_space.sample_randomly(10):
        tic = time.time()
        U = fom.solve(mu)
        t_fom = time.time() - tic

        tic = time.time()
        u_red = rom.solve(mu)
        t_rom = time.time() - tic

        U_red = reductor.reconstruct(u_red)
        errs.append(((U - U_red).l2_norm() / U.l2_norm())[0])
        speedups.append(t_fom / t_rom)
    print(f'Maximum relative ROM error: {max(errs)}')
    print(f'Median of ROM speedup: {np.median(speedups)}')


if __name__ == '__main__':
    args = docopt(__doc__)
    fenics_nonlinear_demo(args)
