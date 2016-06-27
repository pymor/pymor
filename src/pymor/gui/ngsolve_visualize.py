if __name__ == '__main__':
    import os

    from pymor.core.pickle import load

    with open(os.environ['NGSOLVE_VISUALIZE_FILE'], 'rb') as f:
        V, U, legend = load(f)

    grid_functions = []
    for u in U:
        gf = GridFunction(V)
        gf.vec.data = u._list[0].impl
        grid_functions.append(gf)

    for gf, name in zip(grid_functions, legend):
        Draw(gf, V.mesh, name=name)
