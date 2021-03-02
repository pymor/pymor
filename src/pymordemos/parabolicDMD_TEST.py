import matplotlib.pyplot as plt
import numpy as np
import time

from pymor.algorithms.dmd import dmd, rand_dmd, get_amplitudes, get_vandermonde

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

        # advection=LincombFunction([ConstantFunction(np.array([-1., 0]), dim_domain=2)],
        #                           [ProjectionParameterFunctional('speed')]),

        reaction=ConstantFunction(0.5, dim_domain=2),

        # rhs=ExpressionFunction('(x[..., 0] > 0.3) * (x[..., 0] < 0.7) * (x[..., 1] > 0.3)*(x[...,1]<0.7) * 0.',
        #                        dim_domain=2),

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
    diameter=1. / 400.,
    nt=100
)
grid = data['grid']
print(grid)
print()

print('Solve ...')
# U = m.solve({'speed': 0.3})
U = m.solve()

# m.visualize(U, title='Solution of Parabolic Problem')

# ----- Testing DMD -----

"""
Calculate Extrapolation Error
"""

XL = U[:-1]
XR = U[1:]
U_k = U[-1]

alle_fehler_dmd = []
alle_fehler_rdmd = []
alle_fehler_r2dmd = []

alle_zeiten_dmd = []
alle_zeiten_rdmd = []
alle_zeiten_r2dmd = []

Testmoden = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# Testmoden = [1, 2, 3, 4, 5, 10]

for i in Testmoden:
    # dmd
    start_dmd = time.process_time()
    W, E = dmd(A=None, target_rank=i, XL=XL, XR=XR, modes='exact')
    time_dmd = time.process_time() - start_dmd

    Amp_dmd = get_amplitudes(U, W)
    V_dmd = get_vandermonde(U, E)
    B_dmd = np.diag(Amp_dmd)

    U_approx_dmd = W.lincomb(B_dmd).lincomb(V_dmd.T)
    fehler = U - U_approx_dmd
    mse = np.mean((fehler).norm())
    alle_fehler_dmd.append(mse)
    alle_zeiten_dmd.append(time_dmd)
    print('Mean reconstruction Error of Snapshotmatrix - dmd: ', mse)

    # randomized dmd, no oversampling, no power Iterations
    start_rdmd = time.process_time()
    rW, rE = rand_dmd(A=U, target_rank=i, distribution='normal', modes='exact')
    time_rdmd = time.process_time() - start_rdmd

    Amp_rdmd = get_amplitudes(U, rW)
    V_rdmd = get_vandermonde(U, rE)
    B_rdmd = np.diag(Amp_rdmd)

    U_approx_rdmd = rW.lincomb(B_rdmd).lincomb(V_rdmd.T)
    fehler = U - U_approx_rdmd
    mse = np.mean((fehler).norm())
    alle_fehler_rdmd.append(mse)
    alle_zeiten_rdmd.append(time_rdmd)
    print('Mean reconstruction Error of Snapshotmatrix - rand. dmd: ', mse)

    # randomized dmd, oversampling - 5, power Iterations - 2
    start_r2dmd = time.process_time()
    r2W, r2E = rand_dmd(A=U, target_rank=i, distribution='normal', modes='exact', oversampling=10, powerIterations=2)
    time_r2dmd = time.process_time() - start_r2dmd

    Amp_r2dmd = get_amplitudes(U, r2W)
    V_r2dmd = get_vandermonde(U, r2E)
    B_r2dmd = np.diag(Amp_r2dmd)

    U_approx_r2dmd = r2W.lincomb(B_r2dmd).lincomb(V_r2dmd.T)
    fehler = U - U_approx_r2dmd
    mse = np.mean((fehler).norm())
    alle_fehler_r2dmd.append(mse)
    alle_zeiten_r2dmd.append(time_r2dmd)
    print('Mean reconstruction Error of Snapshotmatrix - rand. dmd (ov. & pI): ', mse)


line1, = plt.plot(Testmoden, alle_fehler_dmd, color='blue', marker='x')
line2, = plt.plot(Testmoden, alle_fehler_rdmd, color='orange', marker='x')
line3, = plt.plot(Testmoden, alle_fehler_r2dmd, color='green', marker='x')
plt.title('DMD Reconstruction Error - exact')
plt.ylabel('Error')
plt.xlabel('Number of Modes')
plt.legend((line1, line2, line3), ('dmd', 'rdmd', 'rdmd + optim.'))
plt.show()

line1, = plt.plot(Testmoden, alle_zeiten_dmd, color='blue', marker='x')
line2, = plt.plot(Testmoden, alle_zeiten_rdmd, color='orange', marker='x')
line3, = plt.plot(Testmoden, alle_zeiten_r2dmd, color='green', marker='x')
plt.title('CPU time - exact')
plt.ylabel('Time')
plt.xlabel('Number of Modes')
plt.legend((line1, line2, line3), ('dmd', 'rdmd', 'rdmd + optim.'))
plt.show()
