# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

#!/usr/bin/env python
# coding: utf-8

# # Solving incompressible heat flow in a cavity
# 
# Let us consider the Navier-Stokes equations for the velocity $\mathbf{u}$ and the pressure $p$ of an incompressible fluid
# 
# \begin{align*} 
#     \nabla \cdot \mathbf{u} &= 0,  \\
#     \mathbf{u}_t + \left( \mathbf{u}\cdot\nabla \right)\mathbf{u} + \nabla p - 2\mu \nabla \cdot \mathbf{D}(\mathbf{u}) &= 0,
# \end{align*}
# 
# 
# where  $\mathbf{D}(\mathbf{u}) = \mathrm{sym}(\mathbf{u}) = \frac{1}{2}\left(\nabla \mathbf{u} +  \left( \nabla \mathbf{u} \right)^{\mathrm{T}} \right)$ is the Newtonian fluid's rate of strain tensor and $\mu$ is the viscosity.
# 
# # Versions
# 
# - Python 3.8
# - Pymor 2021
# - Dolfin 2019

# In[2]:


# ### ROM generation (POD/DEIM)
from pymor.algorithms.ei import ei_greedy
from pymor.algorithms.newton import newton
from pymor.algorithms.pod import pod
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.reductors.basic import InstationaryRBReductor
# ### ROM validation
import time
import numpy as np
# ### pyMOR wrapping
from pymor.bindings.fenics import FenicsVectorSpace, FenicsOperator, FenicsVisualizer, FenicsMatrixOperator
from pymor.models.basic import InstationaryModel
from pymor.operators.constructions import VectorOperator
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper


# In[3]:


# FOM
import dolfin as df
# PLOTS
import matplotlib.pyplot as plt


# In[4]:


def plot_w(w, split = False, save = False, outdir = './fig/', name = '', nt = ''):
    
    if (type(nt)!=type('')):
        nt=str(nt)
    
    if split:
        p, u = df.split(w.leaf_node())
    else:
        p, u  = w.split()   
    
    fig = df.plot(u)
    plt.title("Velocity vector field")
    plt.xlabel("$x$")
    plt.ylabel("$y$")    
    plt.colorbar(fig)  
    if save:
        plt.savefig(outdir+name+'velocity'+nt+'.png')
        del fig
        plt.clf()  
        plt.close() 
    else:
        plt.show()
    
    fig = df.plot(p)
    plt.title("Pressure field")
    plt.xlabel("$x$")
    plt.ylabel("$y$")    
    plt.colorbar(fig)    
    if save:
        plt.savefig(outdir+name+'pressure'+nt+'.png')
        del fig
        plt.clf()  
        plt.close() 
    else:
        plt.show()

    if 0:
        fig = df.plot(T)
        plt.title("Temperature field")
        plt.xlabel("$x$")
        plt.ylabel("$y$")    
        plt.colorbar(fig)    
        plt.savefig(outdir+name+'temperature'+nt+'.png')
        del fig
        plt.clf()  
        plt.close() 
    #end
    return


# In[5]:


def discretize(dim, n):
    # ### problem definition

    if dim == 2:
        mesh = df.UnitSquareMesh(n, n)
    else:
        raise NotImplementedError

    P1 = df.FiniteElement('P', mesh.ufl_cell(), 1)
    P2 = df.VectorElement('P', mesh.ufl_cell(), 2, dim = 2)
    # Taylor-Hoods elements
    TH = df.MixedElement([P1, P2])
    W = df.FunctionSpace(mesh, TH)
    W_p = W.sub(0)
    W_u = W.sub(1)

    # define solution vectors and test functions
    p = df.TrialFunction(W_p)
    u = df.TrialFunction(W_u)
    psi_p = df.TestFunction(W_p)
    psi_u = df.TestFunction(W_u)

    # assemble mass matrix
    MASS1 = df.assemble(df.inner(u, psi_u) * df.dx)
    
    fig=plt.spy(MASS1.array())
    plt.show()
  
    # Test functions
    psi_p, psi_u = df.TestFunctions(W)

    # Solution functions
    w = df.Function(W)
    p, u = df.split(w)

    # Parameters
    Re = df.Constant(1.)

    # velocity BCs
    hot_wall    = "near(x[0],  0.)" #x=0 
    cold_wall   = "near(x[0], 1.)" #x= \bar{x}
    top_wall    = "near(x[1], 1.)" #y=1
    bottom_wall = "near(x[1],  0.)" #y=0
    walls = hot_wall + " | " + cold_wall + " | " + bottom_wall
    
    bcu_noslip  = df.DirichletBC(W_u, df.Constant((0, 0)), walls)
    bcu_lid = df.DirichletBC(W_u, df.Constant((1,0)), top_wall)
    
    # pressure BCs
    pressure_point = "near(x[0],  0.) & (x[1]<= "+str(2./n)+")"
    bcp  = df.DirichletBC(W_p, df.Constant(0), pressure_point)
    
    bc = [bcu_noslip, bcu_lid, bcp]

    # define Fenics model
    mass = -psi_p*df.div(u)
    momentum = (df.dot(psi_u, df.dot(df.grad(u), u)) 
                - df.div(psi_u)*p 
                + 2.*(1./Re)*df.inner(df.sym(df.grad(psi_u)), df.sym(df.grad(u))))
    F = (mass+momentum)*df.dx
    
    # solve Full Order Model
    df.solve(F == 0, w, bc,
             solver_parameters={"newton_solver": {"relative_tolerance": 1e-6}})

    plot_w(w)

    # PyMOR binding
    
    # FEM space
    space = FenicsVectorSpace(W)
    # Mass operator
    mass_op = FenicsMatrixOperator(MASS1, W, W, name='mass')
    # Stationary operator
    op = FenicsOperator(F, space, space, w, bc,
                        parameter_setter=lambda mu: Re.assign(mu['Re'].item()),
                        parameters={'Re': 1},
                        solver_options={'inverse': {'type': 'newton', 
                                                    'rtol': 1e-6, 
                                                    'return_residuals': 'True'}})
    

    #time discretization
    nt = 3
    timestep_size = 0.01
    ie_stepper = ImplicitEulerTimeStepper(nt=nt)
    
    # initial conditions
    fom_init = VectorOperator(op.range.zeros()) 
    #rhs
    rhs = VectorOperator(op.range.zeros())
    
    # FOM binding 
    fom = InstationaryModel(timestep_size*nt, 
                            fom_init, 
                            op, 
                            rhs,
                            mass=mass_op,
                            time_stepper=ie_stepper,
                            visualizer=FenicsVisualizer(space))
    
    return fom, W


# In[6]:


def main(n):
    
    """Reduces a FEniCS-based nonlinear diffusion problem using POD/DEIM.
    Input
    - n: Number of mesh intervals per spatial dimension. """
    dim = 2
    
    fom, W = discretize(dim, n)

    # define range for parameters
    parameter_space = fom.parameters.space((1., 500.))
   
    # collect snapshots FOM
    U = fom.solution_space.empty()
    residuals = fom.solution_space.empty()
    for mu in parameter_space.sample_uniformly(10):
        UU = fom.solve(mu)
        U.append(UU)
    #end

    # extract and plot last time step solution
    U_df = df.Function(W)
    U_df.leaf_node().vector()[:] = (U.to_numpy()[-1,:]).squeeze()#.reshape((UU.to_numpy().size,))
    plot_w(U_df, name='fom_')
    
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
        print('error: ', error, 'speedup: ', speedup)
        errs.append(error)
        speedups.append(speedup)
    #endfor
    U_red = df.Function(W)
    U_red.leaf_node().vector()[:] = (U.to_numpy()[-1,:]).squeeze()#.reshape((UU.to_numpy().size,))
    plot_w(U_red, name='rom_')
    
    
    print(f'Maximum relative ROM error: {max(errs)}')
    print(f'Median of ROM speedup: {np.median(speedups)}')


# In[9]:


from dolfin.cpp.parameter import parameters, Parameters
from dolfin.parameter import ffc_default_parameters

if not parameters.has_parameter_set("form_compiler"):
    parameters.add(ffc_default_parameters())



main(10)






