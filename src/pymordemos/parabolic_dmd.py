import matplotlib.pyplot as plt
import numpy as np

from pymor.algorithms.dmd import dmd, rand_dmd

from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ExpressionFunction, ConstantFunction, LincombFunction
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.discretizers.builtin.cg import discretize_instationary_cg
from pymor.discretizers.builtin.grids.tria import TriaGrid

problem = InstationaryProblem(

    StationaryProblem(
        domain=RectDomain(),

        diffusion=ConstantFunction(0.01, dim_domain=2),

        advection=LincombFunction([ConstantFunction(np.array([-1., 0]), dim_domain=2)],
                                  [ProjectionParameterFunctional('speed')]),

        reaction=ConstantFunction(0.5, dim_domain=2),

        rhs=ExpressionFunction('(x[..., 0] > 0.3) * (x[..., 0] < 0.7) * (x[..., 1] > 0.3)*(x[...,1]<0.7) * 0.',
                               dim_domain=2),

        dirichlet_data=ConstantFunction(value=0., dim_domain=2),
    ),

    T=1.,

    initial_data=ExpressionFunction('(x[..., 0] > 0.3) * (x[..., 0] < 0.7) * (x[...,1]>0.3) * (x[..., 1] < 0.7) * 10.',
                                    dim_domain=2),
)

print('Discretize ...')
m, data = discretize_instationary_cg(
    analytical_problem=problem,
    grid_type=TriaGrid,
    diameter=1. / 200.,
    nt=50
)
grid = data['grid']
print(grid)
print()

print('Solve ...')
U = m.solve({'speed': 0.3})
# U = m.solve()

m.visualize(U, title='Solution of Parabolic Problem')


# ----- Testing DMD -----
W, E = dmd(A=U, modes='standard')
W2, E2 = dmd(A=U, modes='exact')
W3, E3 = dmd(A=U, modes='exact_scaled')
W4, E4 = dmd(A=U, modes='standard', order=False)

print('Visualize ...')
m.visualize(W, title='DMD Modes - standard')
plt.plot(E.real, E.imag, 'b.')
plt.title('DMD Eigenvalues - standard')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()

m.visualize(W4, title='DMD Modes - standard - not ordered')
plt.plot(E4.real, E4.imag, 'b.')
plt.title('DMD Eigenvalues - standard - not ordered')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()

m.visualize(W2, title='DMD Modes - exact')
plt.plot(E2.real, E2.imag, 'b.')
plt.title('DMD Eigenvalues - exact')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()

m.visualize(W3, title='DMD Modes - exact_scaled')
plt.plot(E3.real, E3.imag, 'b.')
plt.title('DMD Eigenvalues - exact_scaled')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()


# ----- Testing rDMD -----
rW, rE = rand_dmd(U, None, 1, 'standard', None, 'normal', oversampling=27, powerIterations=3)
rW2, rE2 = rand_dmd(U, None, 1, 'standard', None, 'uniform', oversampling=10, powerIterations=2)
rW3, rE3 = rand_dmd(U, 5, 1, 'exact', None, 'normal', oversampling=10, powerIterations=0)
rW4, rE4 = rand_dmd(U, 12, 1, 'exact', None, 'uniform', oversampling=0, powerIterations=0)

print('Visualize ...')
m.visualize(rW, title='randomized DMD Modes - standard - normal distribution')
plt.plot(rE.real, rE.imag, 'b.')
plt.title('rDMD Eigenvalues - standard - normal distribution')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()

m.visualize(rW2, title='randomized DMD Modes - standard - uniform distribution')
plt.plot(rE2.real, rE2.imag, 'b.')
plt.title('rDMD Eigenvalues - standard - uniform distribution')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()

m.visualize(rW3, title='randomized DMD Modes - exact - normal distribution')
plt.plot(rE3.real, rE3.imag, 'b.')
plt.title('rDMD Eigenvalues - exact - normal distribution')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()

m.visualize(rW4, title='randomized DMD Modes - exact - uniform distribution')
plt.plot(rE4.real, rE4.imag, 'b.')
plt.title('rDMD Eigenvalues - standard - uniform distribution')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()
