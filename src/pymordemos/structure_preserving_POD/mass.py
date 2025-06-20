import numpy as np
import os


def check_sparsity(matrix_array, tolerance):
    zeros = 0
    for row in matrix_array:
        for element in row:
            if np.abs(element) < tolerance:
                zeros += 1
    elements = (matrix_array.shape[0] * matrix_array.shape[1])
    if elements == 0:
        return 0
    else:
        return zeros / elements


sparsity = {}
maximum = {}
absolute_maximum = {}
tolerance = 1e-9
for red_dim in [8, 16, 24, 32, 44, 52, 60]:
    mass = np.loadtxt(f"mass/mass{red_dim}.txt")
    print(mass)
    sparsity[red_dim] = check_sparsity(mass, tolerance)
    maximum[red_dim] = float(np.max(mass))
    absolute_maximum[red_dim] = float(np.max(np.abs(mass)))


print(sparsity)
print(maximum)
print(absolute_maximum)


