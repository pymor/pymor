import numpy as np
from matplotlib import pyplot as plt

# from pymordemos.symplectic_wave_equation import discretize_fom
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.algorithms.pod import pod
from pymor.reductors.basic import InstationaryRBReductor
from pymor.algorithms.projection import project
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import IdentityOperator, LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.models.symplectic import QuadraticHamiltonianModel
from scipy.sparse import diags

from pymor.reductors.reductor_PH import PHReductor, MyQuadraticHamiltonianRBReductor
from pymor.algorithms.PH import POD_PH, check_POD, POD_new

NEW_METHODS = ['POD_PH'] + ['POD_PH_just_Vr'] #+ ['POD_new']
METHODS = NEW_METHODS + ['POD', 'check_POD']


def main(
        final_time: float = 10.,
        rbsize: int = 80,        
):
    fom = discretize_fom(T=final_time)
    # fom = discretize_mass_spring_chain()
    X = fom.solve()
    F = fom.operator.apply(X)
    rel_fac = np.sqrt(X.norm2().sum())

    half_rbsize = min(rbsize // 2, len(X) // 2)
    red_dims = np.linspace(0, half_rbsize, 10, dtype=int)
    for i_red_dim, red_dim in enumerate(red_dims):
        if red_dim % 2 != 0:
            red_dims[i_red_dim] -= 1
    red_dims = red_dims * 2

    results = {}
    for method in METHODS:
        results[method] = run_mor(fom, X, F, method, red_dims)
        

    fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
    markers = {
        'POD_PH': "o",
        'POD': '^',
        'check_POD': 'x',
        'POD_PH_just_Vr': '.',
        'POD_new': ','
    }
    colors = {
        'POD_PH': 'green',
        'POD': 'blue',
        'check_POD': 'pink',
        'POD_PH_just_Vr': 'red',
        'POD_new': 'yellow'
    }
    for method, results in results.items():
        axs[0].semilogy(
            red_dims,
            results['abs_err_proj'] / rel_fac,
            marker=markers[method],
            color=colors[method]
        )
        axs[1].semilogy(
            red_dims,
            results['abs_err_rom'] / rel_fac,
            marker=markers[method],
            color=colors[method],
        )
        axs[2].semilogy(
            red_dims,
            results['abs_err_initial_data'],
            marker=markers[method],
            color=colors[method],
            label=method
        )

    fig.suptitle('All reduction techniques set to PHReductor')
    axs[0].title.set_text('Relative projection error')
    axs[1].title.set_text('Relative reduction error')
    axs[2].title.set_text('Initial data error')
    axs[0].set_xlabel('red. dim. 2k')
    axs[1].set_xlabel('red. dim. 2k')
    axs[0].set_ylabel('rel. err.')

    plt.legend()
    plt.show()




def run_mor(fom, X, F, method, red_dims):
    max_red_dim = red_dims.max()
    if method in NEW_METHODS:
        if method == 'POD_PH':
            max_V_r, max_W_r = POD_PH(X, F, max_red_dim, 2)
            fig1, ax1 = plt.subplots()
        elif method == 'POD_PH_just_Vr':
            max_V_r, max_W_r = POD_PH(X, F, max_red_dim, 2)
        elif method == 'POD_new':
            max_V_r, max_W_r = POD_new(X, max_red_dim, 2)
    else:
        if method == 'check_POD':
            max_V_r = check_POD(X, max_red_dim)
        else:
            max_V_r, svals = pod(X, modes=max_red_dim)
    
    abs_err_proj = np.zeros(len(red_dims))
    abs_err_rom = np.zeros(len(red_dims))
    abs_err_initial_data = np.zeros(len(red_dims))
    for i_red_dim, red_dim in enumerate(red_dims):
        print(method)
        if red_dim > len(max_V_r):
            abs_err_proj[i_red_dim] = np.nan
            abs_err_rom[i_red_dim] = np.nan
            abs_err_initial_data[i_red_dim] = np.nan
            continue
        V_r = max_V_r[:red_dim]
        if method in NEW_METHODS:
            if method == "POD_PH":
                W_r = max_W_r[:red_dim]
                print("POD_PH", len(V_r))
                reductor = MyQuadraticHamiltonianRBReductor(fom, V_r, W_r)
                print(X.dim, len(X), W_r.dim, len(W_r))
                U_proj = V_r.lincomb(W_r.inner(X))
            elif method == 'POD_PH_just_Vr':
                W_r = max_W_r[:red_dim]
                reductor = PHReductor(fom, V_r, V_r)
                U_proj = V_r.lincomb(V_r.inner(X))
            elif method == 'POD_new':
                W_r = max_W_r[:red_dim]
                reductor = PHReductor(fom, V_r, W_r)
                U_proj = V_r.lincomb(W_r.inner(X))
        else:
            if method == "POD":
                V_r = max_V_r[:red_dim]
                W_r = V_r
                reductor = InstationaryRBReductor(fom, V_r)
                U_proj = V_r.lincomb(V_r.inner(X))
            elif method == 'check_POD':
                V_r = max_V_r[:red_dim]
                W_r = V_r
                reductor = PHReductor(fom, V_r, V_r)
                U_proj = V_r.lincomb(V_r.inner(X))
        rom  = reductor.reduce()
        print(len(fom.initial_data.as_vector()), fom.initial_data.as_vector().dim)
        abs_err_initial_data[i_red_dim] = np.sqrt((fom.initial_data.as_vector() - V_r.lincomb(rom.initial_data.as_vector().to_numpy())).norm2())
        u = rom.solve()
        n_x = 500
        wave_speed = 0.1
        l = 1.
        dx = l / (n_x-1)
        if method == 'POD_PH' and red_dim != 0:
            reduced_Hamiltonian = rom.eval_hamiltonian(u)
            ax1.semilogy(reduced_Hamiltonian, label = str(red_dim))
            ax1.semilogy(fom.eval_hamiltonian(X), color = "blue")
            plt.legend()
            plt.title(f"method: {method}, modes: {red_dim}")
            energy = []
            # numpy_u = u.to_numpy()
            # print(np.shape(numpy_u)[0])
            # for i in range(np.shape(numpy_u)[1]):
            #     energy[i] = compute_discretized_Hamiltonian(dx=dx, p=u[], q=u.blocks[1][500:], c=wave_speed)
            #     print(reduced_Hamiltonian - energy)



        
        x_axis = np.arange(0, 1, 0.001)
        numpy_X = (X.blocks[0])[:1000].to_numpy()
        reconstruction = V_r.lincomb(u.to_numpy())
        if red_dim == 60 and method == 'POD_PH':
            fig2, ax2 = plt.subplots()
            numpy_reconstruction = (reconstruction.blocks[0])[:1000].to_numpy()
            for i in range(0, 1002, 150):
                ax2.plot(x_axis, numpy_X[i], color = "red")
                ax2.plot(x_axis, numpy_reconstruction[i], color = "blue")
            plt.ylim(0, 1)
            plt.title(f"method: {method}, number of modes: {red_dim}")
        abs_err_proj[i_red_dim] = np.sqrt((X - U_proj).norm2().sum())
        abs_err_rom[i_red_dim] = np.sqrt((X - reconstruction).norm2().sum())

    return {
        'abs_err_proj': abs_err_proj,
        'abs_err_rom': abs_err_rom,
        'abs_err_initial_data': abs_err_initial_data
    }


def compute_discretized_Hamiltonian(dx, p, q, c):
    n = len(p)
    sum_p = 0
    q_0 = q[n-1]
    sum_q = (q[0] - q_0)**2
    for i in range(n):
        sum_p += p[i]**2
    for i in range(1, n):
        sum_q += (q[i] - q[i-1]) ** 2
    energy = (dx / 2) * sum_p + ((c ** 2) / (2 * dx))
    return energy



def discretize_fom(T=50):
    n_x = 500
    nt = int(T / 0.01) + 1
    wave_speed = 0.1
    l = 1.
    dx = l / (n_x-1)

    # construct H_op
    space = NumpyVectorSpace(n_x)
    Dxx = diags(
        [-2 * np.ones(n_x), np.ones(n_x-1), np.ones(n_x-1)],
        [0, 1, -1],
        format='csr',
    )
    H_op = BlockDiagonalOperator([
        NumpyMatrixOperator(-wave_speed**2 / dx * Dxx),
        LincombOperator([IdentityOperator(space)], [1/dx]),
    ])

    # construct initial_data
    h = lambda s: (0 <= s) * (s <= 1) * (1 - 3/2 * s**2 + 3/4 * s**3) + (1 < s) * (s <= 2) * ((2-s)**3)/4
    bump = lambda xi: h(np.abs(4*(xi - l/2)))
    initial_data = H_op.source.make_array([
        space.make_array(bump(np.linspace(0, l, n_x))),
        space.make_array(np.zeros(n_x)),
    ])
    
    fom = QuadraticHamiltonianModel(T, initial_data, H_op, nt=nt, name='hamiltonian_wave_equation')
    # TODO: fom.operator = fom.operator.with_(solver_options={'type': 'to_matrix'})
    return fom



def discretize_mass_spring_chain(T=10, n=100, k=1.0, m=1.0):
    """
    Discretizes a finite mass-spring chain as a linear Hamiltonian system.

    Parameters:
    - T: final time
    - n: number of masses (degrees of freedom)
    - k: spring stiffness
    - m: mass

    Returns:
    - fom: QuadraticHamiltonianModel
    """
    dt = 0.01
    nt = int(T / dt) + 1

    space = NumpyVectorSpace(n)

    # Mass matrix M and stiffness matrix K (simple 1D second difference with Dirichlet BCs)
    K = diags(
        [2 * np.ones(n), -1 * np.ones(n - 1), -1 * np.ones(n - 1)],
        [0, -1, 1],
        format='csr'
    )

    M_inv = IdentityOperator(space) * (1.0 / m)  # constant mass matrix inverse

    # Construct Hamiltonian operator (block diagonal)
    H_op = BlockDiagonalOperator([
        NumpyMatrixOperator(k * K),  # potential energy (K q)
        M_inv,                       # kinetic energy (M^{-1} p)
    ])

    # Initial data: small displacement in the middle mass, zero momentum
    initial_disp = np.zeros(n)
    initial_disp[n // 2] = 1.0  # impulse at center

    initial_data = H_op.source.make_array([
        space.make_array(initial_disp),   # q
        space.make_array(np.zeros(n)),    # p
    ])

    fom = QuadraticHamiltonianModel(T, initial_data, H_op, nt=nt, name='mass_spring_chain')
    return fom

if __name__ == "__main__":
    main()
        

