if __name__ == '__main__':
    import os

    from pymor.core.pickle import load

    with open(os.environ['NGSOLVE_VISUALIZE_FILE'], 'rb') as f:
        V = load(f)
        vec = load(f)

    u = GridFunction(V)
    u.vec.data = vec

    Draw(u)
