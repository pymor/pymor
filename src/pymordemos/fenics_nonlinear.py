# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typer import Argument, Option, run

from pymor.core.config import config


def main(
    dim: int = Argument(..., help='Spatial dimension of the problem.'),
    n: int = Argument(..., help='Number of mesh intervals per spatial dimension.'),
    order: int = Argument(..., help='Finite element order.'),
    visualize: bool = Option(True, help='Visualize solution and reduczed solution'),
):
    """Reduces a FEniCS-based nonlinear diffusion problem using POD/DEIM."""
    if not config.HAVE_FENICS and not config.HAVE_FENICSX:
        from pymor.core.exceptions import DependencyMissingError
        raise DependencyMissingError('Neither FEniCSx nor legacy FEniCSs have been found.')
    from pymor.tools import mpi

    discretize = discretize_fenicsx if config.HAVE_FENICSX else discretize_fenics

    if mpi.parallel:
        from pymor.models.mpi import mpi_wrap_model
        local_models = mpi.call(mpi.function_call_manage, discretize, dim, n, order)
        fom = mpi_wrap_model(local_models, use_with=True, pickle_local_spaces=False)
    else:
        fom = discretize(dim, n, order)

    parameter_space = fom.parameters.space((0.01, 1000.))

    # ### ROM generation (POD/DEIM)
    from pymor.algorithms.ei import ei_greedy
    from pymor.algorithms.pod import pod
    from pymor.operators.ei import EmpiricalInterpolatedOperator
    from pymor.reductors.basic import StationaryRBReductor
    from pymor.solvers.newton import NewtonSolver

    solver = NewtonSolver(rtol=1e-6, return_residuals=True)
    U = fom.solution_space.empty()
    residuals = fom.solution_space.empty()
    for mu in parameter_space.sample_logarithmic_uniformly(10):
        UU, info = solver.solve(fom.operator, fom.rhs.as_vector(), mu=mu, return_info=True)
        U.append(UU)
        residuals.append(info['residuals'])

    dofs, cb, _ = ei_greedy(residuals, rtol=1e-7)
    ei_op = EmpiricalInterpolatedOperator(fom.operator, collateral_basis=cb, interpolation_dofs=dofs, triangular=True)

    rb, svals = pod(U, rtol=1e-7)
    fom_ei = fom.with_(operator=ei_op)
    reductor = StationaryRBReductor(fom_ei, rb)
    rom = reductor.reduce()
    rom = rom.with_(operator=rom.operator.with_(solver=solver))

    # ### ROM validation
    import time

    import numpy as np

    # ensure that FFC is not called during runtime measurements
    rom.solve(1)

    errs = []
    speedups = []
    mus = parameter_space.sample_logarithmic_randomly(10)
    for mu in mus:
        tic = time.perf_counter()
        U = fom.solve(mu)
        t_fom = time.perf_counter() - tic

        tic = time.perf_counter()
        u_red = rom.solve(mu)
        t_rom = time.perf_counter() - tic

        U_red = reductor.reconstruct(u_red)
        errs.append(((U - U_red).norm() / U.norm())[0])
        speedups.append(t_fom / t_rom)
    print(f'Reduced basis size: {len(rb)}, collateral basis size: {len(cb)}')
    print(f'Maximum relative ROM error: {max(errs)}')
    print(f'Median of ROM speedup: {np.median(speedups)}')

    if visualize:
        mu = mus[np.argmax(errs)]
        U = fom.solve(mu)
        U_red = reductor.reconstruct(rom.solve(mu))
        fom.visualize((U, U_red, U-U_red), legend=('FOM', 'ROM', 'Error'), title=f'c={mu["c"]}')


def discretize_fenics(dim, n, order):
    # ### problem definition
    import dolfin as df

    if dim == 2:
        mesh = df.UnitSquareMesh(n, n)
    elif dim == 3:
        mesh = df.UnitCubeMesh(n, n, n)
    else:
        raise NotImplementedError

    V = df.FunctionSpace(mesh, 'CG', order)

    g = df.Constant(1.0)
    c = df.Constant(1.)

    class DirichletBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - 1.0) < df.DOLFIN_EPS and on_boundary
    db = DirichletBoundary()
    bc = df.DirichletBC(V, g, db)

    u = df.Function(V)
    v = df.TestFunction(V)
    f = df.Expression('x[0]*sin(x[1])', degree=2)
    F = df.inner((1 + c*u**2)*df.grad(u), df.grad(v))*df.dx - f*v*df.dx

    df.solve(F == 0, u, bc,
             solver_parameters={'newton_solver': {'relative_tolerance': 1e-6}})

    # ### pyMOR wrapping
    from pymor.bindings.fenics import FenicsOperator, FenicsVectorSpace, FenicsVisualizer
    from pymor.models.basic import StationaryModel
    from pymor.operators.constructions import VectorOperator

    space = FenicsVectorSpace(V)
    op = FenicsOperator(F, space, space, u, (bc,),
                        parameter_setter=lambda mu: c.assign(mu['c'].item()),
                        parameters={'c': 1})
    rhs = VectorOperator(op.range.zeros())

    fom = StationaryModel(op, rhs,
                          visualizer=FenicsVisualizer(space))

    return fom


def discretize_fenicsx(dim, n, order):
    # ### problem definition
    import numpy as np
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    if dim == 2:
        mesh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    elif dim == 3:
        mesh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    else:
        raise NotImplementedError


    V = fem.functionspace(mesh, ('Lagrange', order))

    g = fem.Constant(mesh, 1.0)
    c = fem.Constant(mesh, 1.0)

    def on_boundary(x):
        return np.isclose(x[0], 1)

    boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)
    bc = fem.dirichletbc(g, boundary_dofs, V)

    u = fem.Function(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = x[0]*ufl.sin(x[1])
    F = ufl.inner((1 + c*u**2)*ufl.grad(u), ufl.grad(v))*ufl.dx - f*v*ufl.dx

    # ### pyMOR wrapping
    from pymor.bindings.fenicsx import FenicsxOperator, FenicsxVectorSpace, FenicsxVisualizer
    from pymor.models.basic import StationaryModel
    from pymor.operators.constructions import VectorOperator

    space = FenicsxVectorSpace(V)
    op = FenicsxOperator(F, u, params={'c': c}, bcs=(bc,), apply_lifting_with_jacobian=True)
    rhs = VectorOperator(op.range.zeros())

    fom = StationaryModel(op, rhs, visualizer=FenicsxVisualizer(space))

    return fom

if __name__ == '__main__':
    run(main)
