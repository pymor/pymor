# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla


def evaluate_bary_a(s, nodes, values, weights, *, weighted_values=None, value_scale=1., derivative=False):
    r"""Evaluate a barycentric rational function in bary-A form or its first derivative.

    The scalar-weight representation of :cite:`GG21` (Section 3) is

    .. math::
        H(s) = \frac{\sum_i w_i F_i/(s-z_i)}{\sum_i w_i/(s-z_i)}.

    Evaluation uses nearest-node rescaling based on :cite:`DZ26` (Algorithm 1).
    At support nodes, :math:`H(z_i)=F_i`. The input arrays must have compatible
    shapes and a common floating-point or complex dtype. They are not modified.

    Parameters
    ----------
    s
        Scalar evaluation point, not NaN. Evaluation at infinity requires a
        nonzero sum of weights.
    nodes
        Nonempty array of distinct finite support nodes, shape `(n,)`.
    values
        Finite sample values as a |NumPy array| of shape `(n,)` or `(n, k)`.
        Flatten matrix samples of shape `(n, p, m)` to `(n, p*m)` before calling.
    weights
        Nonzero scalar weights as a |NumPy array| of shape `(n,)`.
    weighted_values
        Cached products `weights[i] * (values[i] / value_scale)`, with the same shape
        as `values`, or `None`. If `None`, products are formed during evaluation.
        Use `None` if precomputing the products would underflow.
    value_scale
        Positive finite real sample scale used to compute `weighted_values`.
    derivative
        If `True`, evaluate the first derivative instead of the function.

    Returns
    -------
    result
        Function value or first derivative, with the same shape as `values[0]`.

    """
    if len(nodes) == 1:
        _check_point(s)
        return np.zeros_like(values[0]) if derivative else values[0]
    q, qprime, j, at_node = _coefficients(s, nodes, derivative)
    if at_node and not derivative:
        return values[j]
    denominator = q.dot(weights)
    if denominator == 0:
        if not np.isfinite(s):
            raise ValueError('Zero leading denominator at infinity.')
        raise spla.LinAlgError('Singular barycentric denominator.')
    numerator_values = _scale_values(values, value_scale) if weighted_values is None else weighted_values
    if at_node:
        value = _scale_values(values[j], value_scale)
    else:
        coefficients = q * weights if weighted_values is None else q
        value = coefficients.dot(numerator_values)
        value /= denominator
    if derivative:
        dcoefficients = qprime * weights if weighted_values is None else qprime
        dvalue = dcoefficients.dot(numerator_values)
        dvalue -= value * qprime.dot(weights)
        dvalue /= denominator
        value = dvalue
    value *= value_scale
    return value


def evaluate_bary_b(s, nodes, values, weights, *, weighted_values=None, value_scale=1., derivative=False):
    r"""Evaluate a barycentric rational function in left bary-B form or its first derivative.

    The matrix-weight representation of :cite:`GG21` (Section 3) is

    .. math::
        H(s) = \left(\sum_i W_i/(s-z_i)\right)^{-1}
               \left(\sum_i W_i F_i/(s-z_i)\right).

    Evaluation extends the nearest-node rescaling of :cite:`DZ26` to matrix weights.
    At support nodes, :math:`H(z_i)=F_i`. The right form can be evaluated by
    transposing samples and weights, then transposing the result without conjugation.
    The input arrays must have compatible shapes and a common floating-point or
    complex dtype. They are not modified.

    Parameters
    ----------
    s
        Scalar evaluation point, not NaN. Evaluation at infinity requires a
        nonsingular sum of weights.
    nodes
        Nonempty array of distinct finite support nodes, shape `(n,)`.
    values
        Finite sample values as a |NumPy array| of shape `(n, p, m)`.
    weights
        Nonsingular matrix weights as a |NumPy array| of shape `(n, p, p)`.
    weighted_values
        Cached products `weights @ (values / value_scale)` of shape `(n, p, m)`,
        or `None` to compute them during evaluation. Column-major blocks allow
        the contractions to use views without copying.
    value_scale
        Positive finite real sample scale used to compute `weighted_values`.
    derivative
        If `True`, evaluate the first derivative instead of the function.

    Returns
    -------
    result
        Function value or first derivative as a |NumPy array| of shape `(p, m)`.

    """
    if len(nodes) == 1:
        _check_point(s)
        return np.zeros_like(values[0]) if derivative else values[0]
    q, qprime, j, at_node = _coefficients(s, nodes, derivative)
    if at_node and not derivative:
        return values[j]
    n, size = weights.shape[:2]
    # these are views for the model's prearranged column-major blocks. The dot sums
    # over nodes; reshaping and transposing its result gives a column-major LAPACK input.
    flat_weights = weights.swapaxes(1, 2).reshape(n, -1)
    denominator = q.dot(flat_weights).reshape(size, size).T
    try:
        factors = _factorize(denominator, overwrite_a=True)
    except spla.LinAlgError as error:
        if not np.isfinite(s):
            raise ValueError('Singular leading denominator at infinity.') from error
        raise
    if weighted_values is None:
        # standalone callers may omit the cache; repeated model evaluations prepare it once.
        weighted_values = weights @ _scale_values(values, value_scale)
    flat_weighted_values = weighted_values.swapaxes(1, 2).reshape(n, -1)
    if at_node:
        value = _scale_values(values[j], value_scale)
    else:
        numerator = q.dot(flat_weighted_values).reshape(-1, size).T
        value = _solve(factors, numerator)
    if derivative:
        # reuse the value's LU factors for D H' = N' - D' H instead of factorizing D again.
        dN = qprime.dot(flat_weighted_values).reshape(-1, size).T
        dD = qprime.dot(flat_weights).reshape(size, size).T
        dN -= dD @ value
        value = _solve(factors, dN)
    value *= value_scale
    return value


def _check_point(s):
    """Reject nonscalar or NaN evaluation points and return whether the point is infinite."""
    assert getattr(s, 'ndim', None) == 0 or np.isscalar(s) or np.ndim(s) == 0, 's must be a scalar.'
    at_infinity = not np.isfinite(s)
    if at_infinity:
        assert not np.isnan(s), 's must not be NaN.'
    return at_infinity


def _coefficients(s, nodes, derivative):
    """Return the rescaled basis, optional derivative, nearest-node index and exact-hit flag."""
    if _check_point(s):
        q = np.ones(len(nodes), dtype=nodes.dtype)
        return q, np.zeros_like(q) if derivative else None, 0, False
    delta = s - nodes
    j = abs(delta).argmin()
    nearest_delta = delta[j]
    # replace the nearest denominator before division, including exact/subnormal hits.
    delta[j] = 1
    q = nearest_delta / delta
    q[j] = 1
    qprime = None
    if derivative:
        # divide twice instead of squaring the denominator to avoid overflow.
        qprime = (nodes[j] - nodes) / delta
        qprime /= delta
        qprime[j] = 0
    return q, qprime, j, nearest_delta == 0


def _scale_values(array, scale):
    """Divide real and imaginary components separately to handle subnormal real scaling safely."""
    if scale == 1:
        return array
    normalized = np.empty_like(array)
    normalized.real = np.real(array) / scale
    if np.iscomplexobj(array):
        normalized.imag = np.imag(array) / scale
    return normalized


def _normalize(array, label):
    """Return componentwise normalized data and its scale, rejecting loss of nonzero entries."""
    # componentwise scaling avoids overflow in complex magnitudes and subnormal division.
    scale = max(np.max(abs(array.real)), np.max(abs(array.imag)))
    if scale == 0:
        return array.copy(), 1.
    normalized = _scale_values(array, scale)
    if np.any((array != 0) & (normalized == 0)):
        raise ValueError(f'The relative range of {label} is too large for normalization.')
    return normalized, scale


def _factorize(matrix, overwrite_a=False):
    """Compute LU factors with LAPACK, raising on singularity or invalid arguments."""
    # direct LAPACK status codes detect singularity without SciPy's warning-handling path.
    getrf = spla.lapack.zgetrf if matrix.dtype.kind == 'c' else spla.lapack.dgetrf
    lu, pivots, info = getrf(matrix, overwrite_a=overwrite_a)
    if info > 0:
        raise spla.LinAlgError('Singular barycentric denominator.')
    if info < 0:
        raise ValueError(f'Illegal argument {-info} in LAPACK getrf.')
    return lu, pivots


def _solve(factors, rhs):
    """Solve with existing LU factors, overwriting the temporary right-hand side."""
    # each right-hand side is a fresh column-major contraction work array.
    lu, pivots = factors
    getrs = spla.lapack.zgetrs if lu.dtype.kind == 'c' else spla.lapack.dgetrs
    value, info = getrs(lu, pivots, rhs, overwrite_b=True)
    if info != 0:
        raise ValueError(f'Illegal argument {-info} in LAPACK getrs.')
    return value
