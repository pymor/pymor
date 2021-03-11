import matplotlib.pyplot as plt
import numpy as np
import time

from pymor.algorithms.dmd import dmd, rand_dmd, get_amplitudes, get_vandermonde

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

        reaction=ConstantFunction(0.5, dim_domain=2),

        dirichlet_data=ConstantFunction(value=0., dim_domain=2),
    ),

    T=1.,

    initial_data=ExpressionFunction(
        '(x[..., 0] > 0.3) * (x[..., 0] < 0.7) * (x[...,1]>0.3) * (x[..., 1] < 0.7) * 10.', dim_domain=2),
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
U = m.solve()

# m.visualize(U, title='Solution of Parabolic Problem')

# ----- Testing DMD -----

"""
Calculate Extrapolation Error
"""

U_k = U[-1]
U = U[:-1]

alle_fehler_dmd = []
alle_fehler_rdmd = []
alle_fehler_r2dmd = []
alle_fehler_r3dmd = []
alle_fehler_r4dmd = []
alle_fehler_r5dmd = []
alle_fehler_r6dmd = []

ex_fehler_dmd = []
ex_fehler_rdmd = []
ex_fehler_r2dmd = []
ex_fehler_r3dmd = []
ex_fehler_r4dmd = []
ex_fehler_r5dmd = []
ex_fehler_r6dmd = []

alle_zeiten_dmd = []
alle_zeiten_rdmd = []
alle_zeiten_r2dmd = []
alle_zeiten_r3dmd = []
alle_zeiten_r4dmd = []
alle_zeiten_r5dmd = []
alle_zeiten_r6dmd = []

fehler_rdmd_EW = []
fehler_r2dmd_EW = []
fehler_r3dmd_EW = []
fehler_r4dmd_EW = []
fehler_r5dmd_EW = []
fehler_r6dmd_EW = []

# Testmoden = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
Testmoden = [1, 5, 10, 15, 20]
# Testmoden = [1, 2, 3, 4, 5, 10]
# Testmoden = [1, 20, 50, 80]

for i in Testmoden:
    # dmd
    start_dmd = time.process_time()
    W, E = dmd(A=U, target_rank=i, modes='exact')
    time_dmd = time.process_time() - start_dmd

    Amp_dmd = get_amplitudes(U, W)

    sort_idx = np.argsort(np.abs(Amp_dmd))[::-1]
    B_dmd = np.diag(Amp_dmd[sort_idx])
    V_dmd = get_vandermonde(U, E[sort_idx])
    W = W[sort_idx]

    U_approx_dmd = W.lincomb(B_dmd.T).lincomb(V_dmd.T)
    fehler = np.sqrt(np.sum((U - U_approx_dmd).norm2())) / np.sqrt(np.sum(U.norm2()))
    alle_fehler_dmd.append(fehler)
    alle_zeiten_dmd.append(time_dmd)

    Uk_approx_dmd = W.lincomb(B_dmd.T).lincomb(E**(i-1))
    ex_error = (U_k - Uk_approx_dmd).norm() / U_k.norm()
    ex_fehler_dmd.append(ex_error)

    # randomized dmd, no oversampling, no power Iterations
    start_rdmd = time.process_time()
    rW, rE = rand_dmd(A=U, target_rank=i, distribution='normal', modes='exact')
    time_rdmd = time.process_time() - start_rdmd

    Amp_rdmd = get_amplitudes(U, rW)

    sort_idx = np.argsort(np.abs(Amp_rdmd))[::-1]
    B_rdmd = np.diag(Amp_rdmd[sort_idx])
    V_rdmd = get_vandermonde(U, rE[sort_idx])
    rW = rW[sort_idx]

    U_approx_rdmd = rW.lincomb(B_rdmd).lincomb(V_rdmd.T)
    fehler = np.sqrt(np.sum((U_approx_dmd - U_approx_rdmd).norm2())) / np.sqrt(np.sum(U_approx_dmd.norm2()))
    alle_fehler_rdmd.append(fehler)
    alle_zeiten_rdmd.append(time_rdmd)

    Uk_approx_rdmd = rW.lincomb(B_rdmd.T).lincomb(rE**(i-1))
    ex_error = (U_k - Uk_approx_rdmd).norm() / U_k.norm()
    ex_fehler_rdmd.append(ex_error)

    # randomized dmd, oversampling - 1,
    start_r2dmd = time.process_time()
    r2W, r2E = rand_dmd(A=U, target_rank=i, distribution='normal', modes='exact', oversampling=1, powerIterations=0)
    time_r2dmd = time.process_time() - start_r2dmd

    Amp_r2dmd = get_amplitudes(U, r2W)

    sort_idx = np.argsort(np.abs(Amp_r2dmd))[::-1]
    B_r2dmd = np.diag(Amp_r2dmd[sort_idx])
    V_r2dmd = get_vandermonde(U, r2E[sort_idx])
    r2W = r2W[sort_idx]

    U_approx_r2dmd = r2W.lincomb(B_r2dmd).lincomb(V_r2dmd.T)
    fehler = np.sqrt(np.sum((U_approx_dmd - U_approx_r2dmd).norm2())) / np.sqrt(np.sum(U_approx_dmd.norm2()))
    alle_fehler_r2dmd.append(fehler)
    alle_zeiten_r2dmd.append(time_r2dmd)

    Uk_approx_r2dmd = r2W.lincomb(B_r2dmd.T).lincomb(r2E**(i-1))
    ex_error = (U_k - Uk_approx_r2dmd).norm() / U_k.norm()
    ex_fehler_r2dmd.append(ex_error)

    # randomized dmd, oversampling - 2,
    start_r3dmd = time.process_time()
    r3W, r3E = rand_dmd(A=U, target_rank=i, distribution='normal', modes='exact', oversampling=2, powerIterations=0)
    time_r3dmd = time.process_time() - start_r3dmd

    Amp_r3dmd = get_amplitudes(U, r3W)

    sort_idx = np.argsort(np.abs(Amp_r3dmd))[::-1]
    B_r3dmd = np.diag(Amp_r3dmd[sort_idx])
    V_r3dmd = get_vandermonde(U, r3E[sort_idx])
    r3W = r3W[sort_idx]

    U_approx_r3dmd = r3W.lincomb(B_r3dmd).lincomb(V_r3dmd.T)
    fehler = np.sqrt(np.sum((U_approx_dmd - U_approx_r3dmd).norm2())) / np.sqrt(np.sum(U_approx_dmd.norm2()))
    alle_fehler_r3dmd.append(fehler)
    alle_zeiten_r3dmd.append(time_r3dmd)

    Uk_approx_r3dmd = r3W.lincomb(B_r3dmd.T).lincomb(r3E**(i-1))
    ex_error = (U_k - Uk_approx_r3dmd).norm() / U_k.norm()
    ex_fehler_r3dmd.append(ex_error)

    # randomized dmd, oversampling - 4
    start_r4dmd = time.process_time()
    r4W, r4E = rand_dmd(A=U, target_rank=i, distribution='normal', modes='exact', oversampling=4, powerIterations=1)
    time_r4dmd = time.process_time() - start_r4dmd

    Amp_r4dmd = get_amplitudes(U, r4W)

    sort_idx = np.argsort(np.abs(Amp_r4dmd))[::-1]
    B_r4dmd = np.diag(Amp_r4dmd[sort_idx])
    V_r4dmd = get_vandermonde(U, r4E[sort_idx])
    r4W = r4W[sort_idx]

    U_approx_r4dmd = r4W.lincomb(B_r4dmd).lincomb(V_r4dmd.T)
    fehler = np.sqrt(np.sum((U_approx_dmd - U_approx_r4dmd).norm2())) / np.sqrt(np.sum(U_approx_dmd.norm2()))
    alle_fehler_r4dmd.append(fehler)
    alle_zeiten_r4dmd.append(time_r4dmd)

    Uk_approx_r4dmd = r4W.lincomb(B_r4dmd.T).lincomb(r4E**(i-1))
    ex_error = (U_k - Uk_approx_r4dmd).norm() / U_k.norm()
    ex_fehler_r4dmd.append(ex_error)

    # randomized dmd, oversampling - 6
    start_r5dmd = time.process_time()
    r5W, r5E = rand_dmd(A=U, target_rank=i, distribution='normal', modes='exact', oversampling=6, powerIterations=1)
    time_r5dmd = time.process_time() - start_r5dmd

    Amp_r5dmd = get_amplitudes(U, r5W)

    sort_idx = np.argsort(np.abs(Amp_r5dmd))[::-1]
    B_r5dmd = np.diag(Amp_r5dmd[sort_idx])
    V_r5dmd = get_vandermonde(U, r5E[sort_idx])
    r5W = r5W[sort_idx]

    U_approx_r5dmd = r5W.lincomb(B_r5dmd).lincomb(V_r5dmd.T)
    fehler = np.sqrt(np.sum((U_approx_dmd - U_approx_r5dmd).norm2())) / np.sqrt(np.sum(U_approx_dmd.norm2()))
    alle_fehler_r5dmd.append(fehler)
    alle_zeiten_r5dmd.append(time_r5dmd)

    Uk_approx_r5dmd = r5W.lincomb(B_r5dmd.T).lincomb(r5E**(i-1))
    ex_error = (U_k - Uk_approx_r5dmd).norm() / U_k.norm()
    ex_fehler_r5dmd.append(ex_error)

    # randomized dmd, oversampling - 10
    start_r6dmd = time.process_time()
    r6W, r6E = rand_dmd(A=U, target_rank=i, distribution='normal', modes='exact', oversampling=10, powerIterations=2)
    time_r6dmd = time.process_time() - start_r6dmd

    Amp_r6dmd = get_amplitudes(U, r6W)

    sort_idx = np.argsort(np.abs(Amp_r6dmd))[::-1]
    B_r6dmd = np.diag(Amp_r6dmd[sort_idx])
    V_r6dmd = get_vandermonde(U, r6E[sort_idx])
    r6W = r6W[sort_idx]

    U_approx_r6dmd = r6W.lincomb(B_r6dmd).lincomb(V_r6dmd.T)
    fehler = np.sqrt(np.sum((U_approx_dmd - U_approx_r6dmd).norm2())) / np.sqrt(np.sum(U_approx_dmd.norm2()))
    alle_fehler_r6dmd.append(fehler)
    alle_zeiten_r6dmd.append(time_r6dmd)

    Uk_approx_r6dmd = r6W.lincomb(B_r6dmd.T).lincomb(r6E**(i-1))
    ex_error = (U_k - Uk_approx_r6dmd).norm() / U_k.norm()
    ex_fehler_r6dmd.append(ex_error)

    # Difference of first Eigenvalues to dmd
    fehler_rdmd_EW.append(np.abs(E[0]-rE[0]))
    fehler_r2dmd_EW.append(np.abs(E[0]-r2E[0]))
    fehler_r3dmd_EW.append(np.abs(E[0]-r3E[0]))
    fehler_r4dmd_EW.append(np.abs(E[0]-r4E[0]))
    fehler_r5dmd_EW.append(np.abs(E[0]-r5E[0]))
    fehler_r6dmd_EW.append(np.abs(E[0]-r6E[0]))

# visualizing Reconstruction Error
line2, = plt.plot(Testmoden, alle_fehler_rdmd, color='orange', marker='x')
line3, = plt.plot(Testmoden, alle_fehler_r2dmd, color='green', marker='x')
line4, = plt.plot(Testmoden, alle_fehler_r3dmd, color='olive', marker='x')
line5, = plt.plot(Testmoden, alle_fehler_r4dmd, color='yellowgreen', marker='x')
line6, = plt.plot(Testmoden, alle_fehler_r5dmd, color='palegreen', marker='x')
line7, = plt.plot(Testmoden, alle_fehler_r6dmd, color='lime', marker='x')

plt.title('Realtive Error - exact')
plt.ylabel('Realtive error')
plt.xlabel('Number of Modes')
plt.yscale('log')
plt.legend((line2, line3, line4, line5, line6, line7),
           ('rdmd', 'rdmd p=1', 'rdmd p=2', 'rdmd p=4, q=1', 'rdmd p=6, q=1', 'rdmd p=10, q=2'))
plt.show()

# Visualizing Extrapolation Error
line1, = plt.plot(Testmoden, ex_fehler_dmd, color='blue', marker='x')
line2, = plt.plot(Testmoden, ex_fehler_rdmd, color='orange', marker='x')
line3, = plt.plot(Testmoden, ex_fehler_r2dmd, color='green', marker='x')
line4, = plt.plot(Testmoden, ex_fehler_r3dmd, color='olive', marker='x')
line5, = plt.plot(Testmoden, ex_fehler_r4dmd, color='yellowgreen', marker='x')
line6, = plt.plot(Testmoden, ex_fehler_r5dmd, color='palegreen', marker='x')
line7, = plt.plot(Testmoden, ex_fehler_r6dmd, color='lime', marker='x')

plt.title('Extrapolation Error - exact')
plt.ylabel('Error')
plt.xlabel('Number of Modes')
plt.yscale('log')
plt.legend((line1, line2, line3, line4, line5, line6, line7),
           ('dmd', 'rdmd', 'rdmd p=1', 'rdmd p=2', 'rdmd p=4, q=1', 'rdmd p=6, q=1', 'rdmd p=10, q=2'))
plt.show()

# Visualizing CPU time
line1, = plt.plot(Testmoden, alle_zeiten_dmd, color='blue', marker='x')
line2, = plt.plot(Testmoden, alle_zeiten_rdmd, color='orange', marker='x')
line3, = plt.plot(Testmoden, alle_zeiten_r2dmd, color='green', marker='x')
line4, = plt.plot(Testmoden, alle_zeiten_r3dmd, color='olive', marker='x')
line5, = plt.plot(Testmoden, alle_zeiten_r4dmd, color='yellowgreen', marker='x')
line6, = plt.plot(Testmoden, alle_zeiten_r5dmd, color='palegreen', marker='x')
line7, = plt.plot(Testmoden, alle_zeiten_r6dmd, color='lime', marker='x')

plt.title('CPU time - exact')
plt.ylabel('Time')
plt.xlabel('Number of Modes')
plt.legend((line1, line2, line3, line4, line5, line6, line7),
           ('dmd', 'rdmd', 'rdmd p=1', 'rdmd p=2', 'rdmd p=4, q=1', 'rdmd p=6, q=1', 'rdmd p=10, q=2'))
plt.show()

# Visualizing Error of randomized dmd Eigenvalues
line1, = plt.plot(Testmoden, fehler_rdmd_EW, color='orange', marker='x')
line2, = plt.plot(Testmoden, fehler_r2dmd_EW, color='green', marker='x')
line3, = plt.plot(Testmoden, fehler_r3dmd_EW, color='olive', marker='x')
line4, = plt.plot(Testmoden, fehler_r4dmd_EW, color='yellowgreen', marker='x')
line5, = plt.plot(Testmoden, fehler_r5dmd_EW, color='palegreen', marker='x')
line6, = plt.plot(Testmoden, fehler_r6dmd_EW, color='lime', marker='x')

plt.title('Difference of the first Eigenvalue to not rand. DMD ')
plt.ylabel('Absolute Difference')
plt.xlabel('Number of Modes')
plt.yscale('log')
plt.legend((line1, line2, line3, line4, line5, line6),
           ('rdmd', 'rdmd p=1', 'rdmd p=2', 'rdmd p=4, q=1', 'rdmd p=6, q=1', 'rdmd p=10, q=2'))
plt.show()

# Visualizing Error of randomized dmd Eigenvalues of last iteration
# sorted by Dominance of the Eigenvalue
W_np = np.abs(W.real.to_numpy())
rW_np = np.abs(rW.real.to_numpy())
r2W_np = np.abs(r2W.real.to_numpy())
r3W_np = np.abs(r3W.real.to_numpy())
r4W_np = np.abs(r4W.real.to_numpy())
r5W_np = np.abs(r5W.real.to_numpy())
r6W_np = np.abs(r6W.real.to_numpy())

norm_EVECS = np.apply_along_axis(np.linalg.norm, 1, W_np)
norm_rEVECS = np.apply_along_axis(np.linalg.norm, 1, rW_np)
norm_r2EVECS = np.apply_along_axis(np.linalg.norm, 1, r2W_np)
norm_r3EVECS = np.apply_along_axis(np.linalg.norm, 1, r3W_np)
norm_r4EVECS = np.apply_along_axis(np.linalg.norm, 1, r4W_np)
norm_r5EVECS = np.apply_along_axis(np.linalg.norm, 1, r5W_np)
norm_r6EVECS = np.apply_along_axis(np.linalg.norm, 1, r6W_np)

sort_idx = np.argsort(norm_EVECS)[::-1]
EVECS_dmd = norm_EVECS[sort_idx]
EVECS_rdmd = norm_rEVECS[sort_idx]
EVECS_r2dmd = norm_r2EVECS[sort_idx]
EVECS_r3dmd = norm_r3EVECS[sort_idx]
EVECS_r4dmd = norm_r4EVECS[sort_idx]
EVECS_r5dmd = norm_r5EVECS[sort_idx]
EVECS_r6dmd = norm_r6EVECS[sort_idx]

fehler_rdmd = np.abs(EVECS_dmd - EVECS_rdmd)
fehler_r2dmd = np.abs(EVECS_dmd - EVECS_r2dmd)
fehler_r3dmd = np.abs(EVECS_dmd - EVECS_r3dmd)
fehler_r4dmd = np.abs(EVECS_dmd - EVECS_r4dmd)
fehler_r5dmd = np.abs(EVECS_dmd - EVECS_r5dmd)
fehler_r6dmd = np.abs(EVECS_dmd - EVECS_r6dmd)

line1, = plt.plot(fehler_rdmd, color='orange', marker='x')
line2, = plt.plot(fehler_r2dmd, color='green', marker='x')
line3, = plt.plot(fehler_r3dmd, color='olive', marker='x')
line4, = plt.plot(fehler_r4dmd, color='yellowgreen', marker='x')
line5, = plt.plot(fehler_r5dmd, color='palegreen', marker='x')
line6, = plt.plot(fehler_r6dmd, color='lime', marker='x')
plt.title('Difference of the Eigenvectors to not rand. dmd')
plt.ylabel('Mean squared Error')
plt.xlabel('Dominance of Mode')
plt.yscale('log')
plt.legend((line1, line2, line3, line4, line5, line6),
           ('rdmd', 'rdmd p=1', 'rdmd p=2', 'rdmd p=4, q=1', 'rdmd p=6, q=1', 'rdmd p=10, q=2'))
plt.show()
