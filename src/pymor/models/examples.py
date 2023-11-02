# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.core.config import config
from pymor.tools import mpi


def thermal_block_example():
    """Return 2x2 thermal block example.

    Returns
    -------
    fom
        Thermal block problem as a |StationaryModel|.
    """
    from pymor.analyticalproblems.thermalblock import thermal_block_problem
    from pymor.discretizers.builtin import discretize_stationary_cg

    p = thermal_block_problem((2, 2))
    fom, _ = discretize_stationary_cg(p, diameter=1/100)
    return fom


def penzl_example():
    """Return Penzl's example.

    Returns
    -------
    fom
        Penzl's FOM example as an |LTIModel|.
    """
    import numpy as np
    import scipy.sparse as sps

    from pymor.models.iosys import LTIModel

    n = 1006
    A1 = np.array([[-1, 100], [-100, -1]])
    A2 = np.array([[-1, 200], [-200, -1]])
    A3 = np.array([[-1, 400], [-400, -1]])
    A4 = sps.diags(np.arange(-1, -n + 5, -1))
    A = sps.block_diag((A1, A2, A3, A4))
    B = np.ones((n, 1))
    B[:6] = 10
    C = B.T
    fom = LTIModel.from_matrices(A, B, C)

    return fom


def msd_example(n=6, m=2, m_i=4, k_i=4, c_i=1, as_lti=False):
    """Mass-spring-damper model as (port-Hamiltonian) linear time-invariant system.

    Taken from :cite:`GPBV12`.

    Parameters
    ----------
    n
        The order of the model.
    m
        The number or inputs and outputs of the model.
    m_i
        The weight of the masses.
    k_i
        The stiffness of the springs.
    c_i
        The amount of damping.
    as_lti
        If `True`, the matrices of the standard linear time-invariant system are returned.
        Otherwise, the matrices of the port-Hamiltonian linear time-invariant system are returned.

    Returns
    -------
    A
        The LTI |NumPy array| A, if `as_lti` is `True`.
    B
        The LTI |NumPy array| B, if `as_lti` is `True`.
    C
        The LTI |NumPy array| C, if `as_lti` is `True`.
    D
        The LTI |NumPy array| D, if `as_lti` is `True`.
    J
        The pH |NumPy array| J, if `as_lti` is `False`.
    R
        The pH |NumPy array| R, if `as_lti` is `False`.
    G
        The pH |NumPy array| G, if `as_lti` is `False`.
    P
        The pH |NumPy array| P, if `as_lti` is `False`.
    S
        The pH |NumPy array| S, if `as_lti` is `False`.
    N
        The pH |NumPy array| N, if `as_lti` is `False`.
    E
        The LTI |NumPy array| E, if `as_lti` is `True`, or
        the pH |NumPy array| E, if `as_lti` is `False`.
    """
    assert n % 2 == 0
    n //= 2

    A = np.array(
        [[0, 1 / m_i, 0, 0, 0, 0], [-k_i, -c_i / m_i, k_i, 0, 0, 0],
         [0, 0, 0, 1 / m_i, 0, 0], [k_i, 0, -2 * k_i, -c_i / m_i, k_i, 0],
         [0, 0, 0, 0, 0, 1 / m_i], [0, 0, k_i, 0, -2 * k_i, -c_i / m_i]])

    if m == 2:
        B = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]).T
        C = np.array([[0, 1 / m_i, 0, 0, 0, 0], [0, 0, 0, 1 / m_i, 0, 0]])
    elif m == 1:
        B = np.array([[0, 1, 0, 0, 0, 0]]).T
        C = np.array([[0, 1 / m_i, 0, 0, 0, 0]])
    else:
        assert False

    J_i = np.array([[0, 1], [-1, 0]])
    J = np.kron(np.eye(3), J_i)
    R_i = np.array([[0, 0], [0, c_i]])
    R = np.kron(np.eye(3), R_i)

    for i in range(4, n + 1):
        B = np.vstack((B, np.zeros((2, m))))
        C = np.hstack((C, np.zeros((m, 2))))

        J = np.block([
            [J, np.zeros(((i - 1) * 2, 2))],
            [np.zeros((2, (i - 1) * 2)), J_i]
        ])

        R = np.block([
            [R, np.zeros(((i - 1) * 2, 2))],
            [np.zeros((2, (i - 1) * 2)), R_i]
        ])

        A = np.block([
            [A, np.zeros(((i - 1) * 2, 2))],
            [np.zeros((2, i * 2))]
        ])

        A[2 * i - 2, 2 * i - 2] = 0
        A[2 * i - 1, 2 * i - 1] = -c_i / m_i
        A[2 * i - 3, 2 * i - 2] = k_i
        A[2 * i - 2, 2 * i - 1] = 1 / m_i
        A[2 * i - 2, 2 * i - 3] = 0
        A[2 * i - 1, 2 * i - 2] = -2 * k_i
        A[2 * i - 1, 2 * i - 4] = k_i

    Q = spla.solve(J - R, A)
    G = B
    P = np.zeros(G.shape)
    D = np.zeros((m, m))
    E = np.eye(2 * n)
    S = (D + D.T) / 2
    N = -(D - D.T) / 2

    if as_lti:
        return A, B, C, D, E

    return J, R, G, P, S, N, E, Q


def navier_stokes_example(n, nt):
    if mpi.parallel:
        from pymor.models.mpi import mpi_wrap_model
        fom = mpi_wrap_model(lambda: _discretize_navier_stokes(n, nt),
                             use_with=True, pickle_local_spaces=False)
        plot_function = None
    else:
        fom, plot_function = _discretize_navier_stokes(n, nt)
    return fom, plot_function


def _discretize_navier_stokes(n, nt):
    config.require('FENICS')
    import dolfin as df
    import matplotlib.pyplot as plt

    from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
    from pymor.bindings.fenics import FenicsMatrixOperator, FenicsOperator, FenicsVectorSpace, FenicsVisualizer
    from pymor.models.basic import InstationaryModel
    from pymor.operators.constructions import VectorFunctional, VectorOperator

    # create square mesh
    mesh = df.UnitSquareMesh(n, n)

    # create Finite Elements for the pressure and the velocity
    P = df.FiniteElement('P', mesh.ufl_cell(), 1)
    V = df.VectorElement('P', mesh.ufl_cell(), 2, dim=2)
    # create mixed element and function space
    TH = df.MixedElement([P, V])
    W = df.FunctionSpace(mesh, TH)

    # extract components of mixed space
    W_p = W.sub(0)
    W_u = W.sub(1)

    # define trial and test functions for mass matrix
    u = df.TrialFunction(W_u)
    psi_u = df.TestFunction(W_u)

    # assemble mass matrix for velocity
    mass_mat = df.assemble(df.inner(u, psi_u) * df.dx)

    # define trial and test functions
    psi_p, psi_u = df.TestFunctions(W)
    w = df.Function(W)
    p, u = df.split(w)

    # set Reynolds number, which will serve as parameter
    Re = df.Constant(1.)

    # define walls
    top_wall = 'near(x[1], 1.)'
    walls = 'near(x[0], 0.) | near(x[0], 1.) | near(x[1], 0.)'

    # define no slip boundary conditions on all but the top wall
    bcu_noslip_const = df.Constant((0., 0.))
    bcu_noslip = df.DirichletBC(W_u, bcu_noslip_const, walls)
    # define Dirichlet boundary condition for the velocity on the top wall
    bcu_lid_const = df.Constant((1., 0.))
    bcu_lid = df.DirichletBC(W_u, bcu_lid_const, top_wall)

    # fix pressure at a single point of the domain to obtain unique solutions
    pressure_point = 'near(x[0],  0.) & (x[1] <= ' + str(2./n) + ')'
    bcp_const = df.Constant(0.)
    bcp = df.DirichletBC(W_p, bcp_const, pressure_point)

    # collect boundary conditions
    bc = [bcu_noslip, bcu_lid, bcp]

    mass = -psi_p * df.div(u)
    momentum = (df.dot(psi_u, df.dot(df.grad(u), u))
                - df.div(psi_u) * p
                + 2.*(1./Re) * df.inner(df.sym(df.grad(psi_u)), df.sym(df.grad(u))))
    F = (mass + momentum) * df.dx

    df.solve(F == 0, w, bc)

    # define pyMOR operators
    space = FenicsVectorSpace(W)
    mass_op = FenicsMatrixOperator(mass_mat, W, W, name='mass')
    op = FenicsOperator(F, space, space, w, bc,
                        parameter_setter=lambda mu: Re.assign(mu['Re'].item()),
                        parameters={'Re': 1})

    # timestep size for the implicit Euler timestepper
    dt = 0.01
    ie_stepper = ImplicitEulerTimeStepper(nt=nt)

    # define initial condition and right hand side as zero
    fom_init = VectorOperator(op.range.zeros())
    rhs = VectorOperator(op.range.zeros())
    # define output functional
    output_func = VectorFunctional(op.range.ones())

    # construct instationary model
    fom = InstationaryModel(dt * nt,
                            fom_init,
                            op,
                            rhs,
                            mass=mass_op,
                            time_stepper=ie_stepper,
                            output_functional=output_func,
                            visualizer=FenicsVisualizer(space))

    def plot_fenics(w, title=''):
        v = df.Function(W)
        v.leaf_node().vector()[:] = (w.to_numpy()[-1, :]).squeeze()
        p, u = v.split()

        fig_u = df.plot(u)
        plt.title('Velocity vector field ' + title)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar(fig_u)
        plt.show()

        fig_p = df.plot(p)
        plt.title('Pressure field ' + title)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar(fig_p)
        plt.show()

    if mpi.parallel:
        return fom
    else:
        return fom, plot_fenics
