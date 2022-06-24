# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Incompressible heat flow in a cavity.

Let us consider the Navier-Stokes equations for the velocity $\mathbf{u}$ and the pressure $p$ of an incompressible fluid

\begin{align*}
    \nabla \cdot \mathbf{u} &= 0,  \\
    \mathbf{u}_t + \left( \mathbf{u}\cdot\nabla \right)\mathbf{u} + \nabla p - 2\mu \nabla \cdot \mathbf{D}(\mathbf{u}) &= 0,
\end{align*}


where $\mathbf{D}(\mathbf{u}) = \mathrm{sym}(\mathbf{u}) = \frac{1}{2}\left(\nabla \mathbf{u} +  \left( \nabla \mathbf{u} \right)^{\mathrm{T}} \right)$ is the Newtonian fluid's rate of strain tensor and $\mu$ is the viscosity.
"""

from typer import Option, run

# ### ROM generation (POD)
from pymor.algorithms.pod import pod
from pymor.reductors.basic import InstationaryRBReductor
# ### ROM validation
import time
import numpy as np
# ### pyMOR wrapping
from pymor.bindings.fenics import FenicsVectorSpace, FenicsOperator, FenicsVisualizer, FenicsMatrixOperator
from pymor.models.basic import InstationaryModel
from pymor.operators.constructions import VectorOperator
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper

import dolfin as df
import matplotlib.pyplot as plt


def plot(w, title_prefix=''):
    p, u  = w.split()

    fig = df.plot(u)
    plt.title("Velocity vector field " + title_prefix)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.colorbar(fig)

    plt.show()

    fig = df.plot(p)
    plt.title("Pressure field " + title_prefix)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.colorbar(fig)

    plt.show()


def discretize(n):
    mesh = df.UnitSquareMesh(n, n)

    P1 = df.FiniteElement('P', mesh.ufl_cell(), 1)
    P2 = df.VectorElement('P', mesh.ufl_cell(), 2, dim=2)
    TH = df.MixedElement([P1, P2])
    W = df.FunctionSpace(mesh, TH)

    W_p = W.sub(0)
    W_u = W.sub(1)

    p = df.TrialFunction(W_p)
    u = df.TrialFunction(W_u)
    psi_p = df.TestFunction(W_p)
    psi_u = df.TestFunction(W_u)

    mass_mat = df.assemble(df.inner(u, psi_u) * df.dx)

    psi_p, psi_u = df.TestFunctions(W)

    w = df.Function(W)
    p, u = df.split(w)

    Re = df.Constant(1.)

    top_wall = "near(x[1], 1.)"
    walls = "near(x[0], 0.) | near(x[0], 1.) | near(x[1], 0.)"

    bcu_noslip_const = df.Constant((0., 0.))
    bcu_noslip  = df.DirichletBC(W_u, bcu_noslip_const, walls)
    bcu_lid_const = df.Constant((1., 0.))
    bcu_lid = df.DirichletBC(W_u, bcu_lid_const, top_wall)

    pressure_point = "near(x[0],  0.) & (x[1]<= " + str(2./n) + ")"
    bcp_const = df.Constant(0.)
    bcp  = df.DirichletBC(W_p, bcp_const, pressure_point)

    bc = [bcu_noslip, bcu_lid, bcp]

    mass = -psi_p * df.div(u)
    momentum = (df.dot(psi_u, df.dot(df.grad(u), u))
                - df.div(psi_u) * p
                + 2.*(1./Re) * df.inner(df.sym(df.grad(psi_u)), df.sym(df.grad(u))))
    F = (mass + momentum) * df.dx

    df.solve(F == 0, w, bc, solver_parameters={"newton_solver": {"relative_tolerance": 1e-6}})

    space = FenicsVectorSpace(W)
    mass_op = FenicsMatrixOperator(mass_mat, W, W, name='mass')
    op = FenicsOperator(F, space, space, w, bc,
                        parameter_setter=lambda mu: Re.assign(mu['Re'].item()),
                        parameters={'Re': 1},
                        solver_options={'inverse': {'type': 'newton',
                                                    'rtol': 1e-6,
                                                    'return_residuals': 'True'}})

    nt = 5
    timestep_size = 0.01
    ie_stepper = ImplicitEulerTimeStepper(nt=nt)

    fom_init = VectorOperator(op.range.zeros())
    rhs = VectorOperator(op.range.zeros())

    fom = InstationaryModel(timestep_size * nt,
                            fom_init,
                            op,
                            rhs,
                            mass=mass_op,
                            time_stepper=ie_stepper,
                            visualizer=FenicsVisualizer(space))

    return fom, W


def main(n: int = Option(10, help='Number of mesh intervals per spatial dimension.')):
    """Reduces a FEniCS-based incompressible Navier-Stokes problem using POD."""
    fom, W = discretize(n)

    # define range for parameters
    parameter_space = fom.parameters.space((1., 50.))

    # collect snapshots FOM
    U = fom.solution_space.empty()
    for mu in parameter_space.sample_uniformly(20):
        UU = fom.solve(mu)
        U.append(UU)

    # build reduced basis
    rb, svals = pod(U, rtol=1e-7)
    reductor = InstationaryRBReductor(fom, rb)
    rom = reductor.reduce()
    # the reductor currently removes all solver_options so we need to add them again
    rom = rom.with_(operator=rom.operator.with_(solver_options=fom.operator.solver_options))

    # ensure that FFC is not called during runtime measurements
    rom.solve(1)

    # validate ROM
    errs = []
    speedups = []
    for mu in parameter_space.sample_randomly(10):
        tic = time.perf_counter()
        U = fom.solve(mu)
        t_fom = time.perf_counter() - tic

        tic = time.perf_counter()
        u_red = rom.solve(mu)
        t_rom = time.perf_counter() - tic

        U_red = reductor.reconstruct(u_red)
        diff = U - U_red
        error = np.linalg.norm(diff.to_numpy())/np.linalg.norm(U.to_numpy())
        speedup = t_fom / t_rom

        errs.append(error)
        speedups.append(speedup)

    U_df = df.Function(W)
    U_df.leaf_node().vector()[:] = (U.to_numpy()[-1, :]).squeeze()
    plot(U_df, title_prefix='FOM')

    U_red_df = df.Function(W)
    U_red_df.leaf_node().vector()[:] = (U_red.to_numpy()[-1, :]).squeeze()
    plot(U_red_df, title_prefix='ROM')

    print(f'Mean of relative ROM error: {np.mean(errs)}')
    print(f'Median of ROM speedup: {np.median(speedups)}')


if __name__ == '__main__':
    run(main)
