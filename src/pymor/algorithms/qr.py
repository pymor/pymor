from pymor.algorithms.cholesky_qr import cholesky_qr
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_vec, block_gram_schmidt


def qr(V, solver='gram_schmidt', **solver_kwargs):
    if solver == 'gram_schmidt':
        return gram_schmidt(V, **solver_kwargs)
    elif solver == 'gram_schmidt_vec':
        return gram_schmidt_vec(V, **solver_kwargs)
    elif solver == 'cholesky_qr':
        return cholesky_qr(V, **solver_kwargs)
    elif solver == 'block_gram_schmidt':
        return block_gram_schmidt(V, **solver_kwargs)
