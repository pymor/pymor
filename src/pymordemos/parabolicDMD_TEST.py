import matplotlib.pyplot as plt
import numpy as np

from pymor.algorithms.dmd import dmd, get_amplitudes, get_vandermonde

from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ExpressionFunction, ConstantFunction
from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.discretizers.builtin.cg import discretize_instationary_cg
from pymor.discretizers.builtin.grids.tria import TriaGrid

problem = InstationaryProblem(

    StationaryProblem(
        domain=RectDomain(),

        diffusion=ConstantFunction(0.01, dim_domain=2),

        # advection=LincombFunction([ConstantFunction(np.array([-1., 0]), dim_domain=2)],
        #                           [ProjectionParameterFunctional('speed')]),

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
# U = m.solve({'speed': 0.3})
U = m.solve()

m.visualize(U, title='Solution of Parabolic Problem')

# ----- Testing DMD -----

"""
Calculate Extrapolation Error
"""
XL = U[:-1]
XR = U[1:]
U_k = U[-1]

WF, EF = dmd(A=None, XL=XL, XR=XR, modes='exact')
# W, E = dmd(A=U, modes='exact')


Amp = get_amplitudes(U, WF)
# print(Amp)
# print('Amp: ', Amp.shape)
V = get_vandermonde(U, EF)
# print(V)
# print('V: ', V.shape)

t = len(U)

"""
Calculation Reconstruction Error
"""
B = np.diag(Amp)
# print(B)
# print('B: ', B.shape)
U_approx = WF.lincomb((B @ V).T)

fehler = U.__sub__(U_approx)
for i in range(len(fehler)):
    print(fehler[i].amax()[1])

mse = np.mean((fehler).norm())
print('Mean reconstruction Error of Snapshotmatrix: ', mse)

m.visualize(U_approx, title='Approximation of Parabolic Problem')

print('Visualize ...')
m.visualize(WF, title='DMD Modes - exact')
plt.plot(EF.real, EF.imag, 'b.')
plt.title('DMD Eigenvalues - exact')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()
