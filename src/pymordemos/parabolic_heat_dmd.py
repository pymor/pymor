# import numpy as np
import matplotlib.pyplot as plt

from pymor.algorithms.dmd import dmd, rand_dmd

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction, LincombFunction
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.core.logger import set_log_levels
from pymor.discretizers.builtin import discretize_instationary_cg
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.discretizers.builtin.grids.tria import TriaGrid


"""Parametric 1D heat equation example."""
set_log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING'})

# Model
p = InstationaryProblem(
    StationaryProblem(
        domain=RectDomain(([0, 0], [1, 1]), left='robin', right='robin', top='robin', bottom='robin'),
        diffusion=LincombFunction([ExpressionFunction('(x[...,0] <= 0.5) * 1.', 2),
                                  ExpressionFunction('(0.5 < x[...,0]) * 1.', 2)],
                                  [2,
                                  ProjectionParameterFunctional('diffusion')]),
        # diffusion=ConstantFunction(0.01, dim_domain=2),
        robin_data=(ConstantFunction(1., 2), ExpressionFunction('(x[...,0] < 1e-10) * 1.', 2)),
        outputs=(('l2_boundary', ExpressionFunction('(x[...,0] > (1 - 1e-10)) * 1.', 2)),),
    ),
    ConstantFunction(0., 2),
    T=5.
)

m, data = discretize_instationary_cg(
    analytical_problem=p,
    grid_type=TriaGrid,
    diameter=1./300.,
    nt=100
)

grid = data['grid']
print(grid)
print()

U1 = m.solve(mu=0.1)
U2 = m.solve(mu=1)
U3 = m.solve(mu=10)

m.visualize(U1, title='Solution of Parabolic Heat Problem - mu=0.1')
m.visualize(U2, title='Solution of Parabolic Heat Problem - mu=1')
m.visualize(U3, title='Solution of Parabolic Heat Problem - mu=10')

W, E = dmd(A=U2, modes='standard')
W2, E2 = dmd(A=U2, modes='exact')
W3, E3 = dmd(A=U2, modes='exact_scaled')
W4, E4 = dmd(A=U2, modes='standard', order=False)

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
rW, rE = rand_dmd(U2, None, 1, 'standard', None, 'normal', oversampling=5, powerIterations=1)
rW2, rE2 = rand_dmd(U2, None, 1, 'standard', None, 'uniform', oversampling=10, powerIterations=2)
rW3, rE3 = rand_dmd(U2, None, 1, 'exact', None, 'normal', oversampling=5, powerIterations=2)
rW4, rE4 = rand_dmd(U2, None, 1, 'exact', None, 'uniform', oversampling=5, powerIterations=2)

print('Visualize ...')
m.visualize(rW, title='randomized DMD Modes - standard - normal distribution')
plt.plot(rE.real, rE.imag, 'b.')
plt.plot(E.real, E.imag, 'rx')
plt.title('rDMD Eigenvalues - standard - normal distribution')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()

m.visualize(rW2, title='randomized DMD Modes - standard - uniform distribution')
plt.plot(rE2.real, rE2.imag, 'b.')
plt.plot(E.real, E.imag, 'rx')
plt.title('rDMD Eigenvalues - standard - uniform distribution')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()

m.visualize(rW3, title='randomized DMD Modes - exact - normal distribution')
plt.plot(rE3.real, rE3.imag, 'b.')
plt.plot(E2.real, E2.imag, 'rx')
plt.title('rDMD Eigenvalues - exact - normal distribution')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()

m.visualize(rW4, title='randomized DMD Modes - exact - uniform distribution')
plt.plot(rE4.real, rE4.imag, 'b.')
plt.plot(E2.real, E2.imag, 'rx')
plt.title('rDMD Eigenvalues - standard - uniform distribution')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()
