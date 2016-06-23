if __name__ == '__main__':
    import os

    from pymor.core.pickle import load

    with open(os.environ['NGSOLVE_VISUALIZE_FILE'], 'rb') as f:
        u = load(f)

    Draw(u)
