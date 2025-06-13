import numpy as np

NEW_METHODS = ['POD_PH'] + ['POD_PH_just_Vr'] #+ ['POD_new']
METHODS = NEW_METHODS + ['POD', 'check_POD']

red_dims = np.loadtxt(f"results/red_dims.txt")
print(red_dims)
for method in METHODS:
    for dim in [0, 8, 16, 24, 32, 44, 52, 60]:
        H = np.loadtxt(f"results/Hamiltonian_reconstruction_{method}_{dim}.txt")
        time = np.loadtxt(f"results/time.txt")
        print(H.shape[0])
        print(time.shape[0])
        data = np.column_stack((time, H))
        np.savetxt(f"results/hamiltonian_plot_{method}_{dim}.txt", data, header="time H", comments='')
