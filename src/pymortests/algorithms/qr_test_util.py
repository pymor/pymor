from scipy.linalg import hilbert

from pymor.vectorarrays.interface import VectorArray, VectorSpace


def generate_hilbert_va(vector_space_type: VectorSpace, n: int) -> VectorArray:
    """Generates a Hilbert matrix as a VectorArray of the specified VectorSpace type.

    Creates a in exact arithmetic full-rank, potentially ill-conditioned,
    square Hilbert matrix of the given dimension `n`.
    Larger hilbert matrices are more ill-conditioned,
    but are never rank-deficient in exact arithmetic
    """
    return vector_space_type(n).from_numpy(hilbert(n))
