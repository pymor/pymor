# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Matrix-free |Operators| represented by |NumPy| arrays with DFT-based matrix-vector multiplicaion.

This module provides the following |NumPy|-based |Operators|:

- |DFTBasedOperator| should be used as a base class for all |Operators| that use DFT matvecs.
- |ToeplitzOperator| matrix-free operator of a Toeplitz matrix.
- |CirculantOperator| matrix-free operator of a circulant matrix.
- |HankelOperator| matrix-free operator of a Hankel matrix.
"""

import numpy as np
from numpy.fft import fft, ifft, irfft, rfft

from pymor.core.cache import CacheableObject, cached
from pymor.operators.interface import Operator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class DFTBasedOperator(Operator, CacheableObject):
    """Base class for operators whose `apply` can be expressed as a DFT operation.

    Implements efficient matrix-vector multiplications with DFT for matrix-free operations and
    caches the circulant vector in the frequency domain.

    Parameters
    ----------
    _arr
        The one-dimensional |NumPy array| that defines the circulant operator.
    source_id
        The id of the operator's `source` |VectorSpace|.
    range_id
        The id of the operator's `range` |VectorSpace|.
    name
        Name of the operator.
    """

    cache_region = 'memory'

    def __init__(self, _arr, source_id=None, range_id=None, name=None):
        _arr = np.squeeze(_arr)
        assert _arr.ndim == 1
        _arr.setflags(write=False)  # make numpy arrays read-only
        self.__auto_init(locals())
        self.linear = True

    @cached
    def _circulant(self):
        return (rfft(self._arr, axis=0) if np.isrealobj(self._arr) else fft(self._arr, axis=0))[:, np.newaxis]

    def _circulant_matvec(self, vec):
        n = vec.shape[0]
        # use real arithmetic if possible
        if np.isrealobj(self._arr) and np.isrealobj(vec):
            F, iF = lambda x: rfft(x, axis=0), lambda x: irfft(x, axis=0, n=n)
        else:
            F, iF = lambda x: fft(x, axis=0), lambda x: ifft(x, axis=0)
        C = self._circulant()
        if np.isrealobj(self._arr) and np.iscomplexobj(vec):
            C = np.concatenate([C, C[1:(None if n % 2 else -1)].conj()[::-1]])

        return iF(F(vec)*C)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        return self.H.apply(V, mu=mu)


class ToeplitzOperator(DFTBasedOperator):
    r"""Matrix-free representation of a Toeplitz matrix by a |NumPy Array|.

    A Toeplitz matrix is a matrix with constant diagonals, i.e.:

    .. math::
        T =
            \begin{bmatrix}
                c_1 & r_2 & r_3 & \cdots & \cdots & r_n \\
                c_2 & c_1 & r_2&&& \vdots\\
                c_3 & c_2&\ddots&&&\vdots\\
                \vdots &&&\ddots&r_2&r_3\\
                \vdots &&&c_2& c_1 & r_2\\
                c_m & \cdots & \cdots & c_3 & c_2& c_1
            \end{bmatrix}\in\mathbb{C}^{m\times n},

    where :math:`c\in\mathbb{C}^m` and :math:`r\in\mathbb{C}^n` denote the first column and first
    row of the Toeplitz matrix, respectively. The matrix :math:`T` as seen above is not explicitly
    constructed, only the arrays `c` and `r` are stored. The operator's `apply` method takes
    advantage of the fact that any Toeplitz matrix can be embedded in a larger circulant matrix to
    leverage efficient matrix-vector multiplications with DFT.

    Parameters
    ----------
    c
        The |NumPy array| that defines the first column of the Toeplitz matrix.
    r
        The |NumPy array| that defines the first row of the Hankel matrix. If supplied, its first
        entry `r[0]` has to equal to `c[0]`. Defaults to `None`. If `r` is `None`, the behaviour
        of scipy.linalg.toeplitz is mimicked which sets `r = c.conj()` (except for the first entry).
    source_id
        The id of the operator's `source` |VectorSpace|.
    range_id
        The id of the operator's `range` |VectorSpace|.
    name
        Name of the operator.
    """

    def __init__(self, c, r=None, source_id=None, range_id=None, name=None):
        c = np.squeeze(c)
        assert c.ndim == 1
        if r is None:
            r = c.conj()
            r[0] = c[0]
        else:
            r = np.squeeze(r)
            assert r.ndim == 1
            assert r[0] == c[0]
        c.setflags(write=False)
        r.setflags(write=False)
        super().__init__(np.concatenate([c, r[:0:-1]]), source_id=source_id, range_id=range_id, name=name)
        self.c = c
        self.r = r
        self.source = NumpyVectorSpace(r.size, source_id)
        self.range = NumpyVectorSpace(c.size, range_id)

    def apply(self, U, mu=None):
        assert U in self.source
        U = np.concatenate([U.to_numpy().T, np.zeros((self._arr.size - U.dim, len(U)))])
        return self.range.make_array(self._circulant_matvec(U)[:self.range.dim].T)

    @property
    def H(self):
        return self.with_(c=self.r.conj(), r=self.c.conj(), source_id=self.range_id, range_id=self.source_id,
                          name=self.name + '_adjoint')


class CirculantOperator(ToeplitzOperator):
    r"""Matrix-free representation of a circulant matrix by a |NumPy Array|.

    A circulant matrix is a special kind of Toeplitz matrix which is square and completely
    determined by its first column via

    .. math::
        C =
            \begin{bmatrix}
                c_1 & c_n & c_{n-1} & \cdots & \cdots & c_2 \\
                c_2 & c_1 & c_n&&& \vdots\\
                c_3 & c_2&\ddots&&&\vdots\\
                \vdots &&&\ddots&c_n&c_{n-1}\\
                \vdots &&&c_2& c_1 & c_n\\
                c_n & \cdots & \cdots & c_3 & c_2& c_1
            \end{bmatrix}\in\mathbb{C}^{n\times n},

    where the so-called circulant vector :math:`c\in\mathbb{C}^n` denotes the first column of the
    matrix. The matrix :math:`C` as seen above is not explicitly constructed, only `c` is stored.
    Efficient matrix-vector multiplications are realized with DFT in the class' `apply` method.
    See :cite:`GVL13` Chapter 4.8.2. for details.

    Parameters
    ----------
    c
        The |NumPy array| that defines the circulant vector.
    source_id
        The id of the operator's `source` |VectorSpace|.
    range_id
        The id of the operator's `range` |VectorSpace|.
    name
        Name of the operator.
    """

    def __init__(self, c, source_id=None, range_id=None, name=None):
        c = np.squeeze(c)
        assert c.ndim == 1
        c.setflags(write=False)
        super(ToeplitzOperator, self).__init__(c, source_id=source_id, range_id=range_id, name=name)
        self.c = self._arr
        self.source = NumpyVectorSpace(c.size, source_id)
        self.range = NumpyVectorSpace(c.size, range_id)

    def apply(self, U, mu=None):
        assert U in self.source
        U = U.to_numpy().T
        return self.range.make_array(self._circulant_matvec(U).T)

    @property
    def H(self):
        return self.with_(c=np.roll(self.c.conj(), -1, axis=0)[::-1],
                          source_id=self.range_id, range_id=self.source_id, name=self.name + '_adjoint')


class HankelOperator(DFTBasedOperator):
    r"""Matrix-free representation of a Hankel matrix by a |NumPy Array|.

    A Hankel matrix is a matrix with constant anti-diagonals, i.e.:

    .. math::
        H =
            \begin{bmatrix}
                c_1 & c_2 & c_3 & \cdots & \cdots & r_1 \\
                c_2 & c_3 & &&& \vdots\\
                c_3 &&&&&\vdots\\
                \vdots &&&&&r_{n-2}\\
                \vdots &&&& r_{n-2} & r_{n-1}\\
                c_m & \cdots & \cdots & r_{n-2} & r_{n-1}& r_n
            \end{bmatrix}\in\mathbb{C}^{m\times n},

    where :math:`c\in\mathbb{C}^m` and :math:`r\in\mathbb{C}^n` denote the first column and last
    row of the Hankel matrix, respectively.
    The matrix :math:`H` as seen above is not explicitly constructed, only the arrays `c` and `r`
    are stored. Efficient matrix-vector multiplications are realized with DFT in the class' `apply`
    method (see :cite:`MSKC21` Algorithm 3.1. for details).

    Parameters
    ----------
    c
        The |NumPy array| that defines the first column of the Hankel matrix.
    r
        The |NumPy array| that defines the last row of the Hankel matrix. If supplied, its first
        entry `r[0]` has to equal to `c[-1]`. Defaults to `None`. If `r` is `None`, the behaviour
        of scipy.linalg.hankel is mimicked which sets `r` to zero (except for the first entry).
    source_id
        The id of the operator's `source` |VectorSpace|.
    range_id
        The id of the operator's `range` |VectorSpace|.
    name
        Name of the operator.
    """

    def __init__(self, c, r=None, source_id=None, range_id=None, name=None):
        c = np.squeeze(c)
        assert c.ndim == 1
        if r is None:
            r = np.zeros_like(c)
            r[0] = c[-1]
        else:
            r = np.squeeze(r)
            assert r.ndim == 1
            assert r[0] == c[-1]
        c.setflags(write=False)
        r.setflags(write=False)
        n = c.size + r.size - 1
        h = np.concatenate([c, r[1:], np.zeros([1 - n % 2])])
        shift = n // 2 + int(np.ceil((c.size - r.size) / 2)) + 1  # this works
        super().__init__(np.roll(h, shift), source_id=source_id, range_id=range_id, name=name)
        self.source = NumpyVectorSpace(r.size, source_id)
        self.range = NumpyVectorSpace(c.size, range_id)
        self.c = c
        self.r = r

    def apply(self, U, mu=None):
        assert U in self.source
        U = U.to_numpy().T
        n = self.c.size + self.r.size - 1
        x = np.concatenate([np.flip(U, axis=0), np.zeros((self.c.size - n % 2, U.shape[1]))])
        return self.range.make_array(self._circulant_matvec(x)[:self.range.dim].T)

    @property
    def H(self):
        h = np.concatenate([self.c, self.r[1:]]).conj()
        return self.with_(c=h[:self.source.dim], r=h[self.source.dim-1:],
                          source_id=self.range_id, range_id=self.source_id, name=self.name+'_adjoint')


class BlockDFTBasedOperator(DFTBasedOperator):
    cache_region = None

    def __init__(self, _ops, source_id=None, range_id=None, name=None):
        assert _ops.ndim == 2
        _ops.setflags(write=False)
        self.__auto_init(locals())
        p, m = self._ops.shape
        source_dim = m * np.max([op.source.dim for op in self._ops.ravel()])
        range_dim = p * np.max([op.range.dim for op in self._ops.ravel()])
        self.source = NumpyVectorSpace(source_dim, source_id)
        self.range = NumpyVectorSpace(range_dim, range_id)
        self.linear = all([op.linear for op in self._ops.ravel()])

    def apply(self, U, mu=None):
        U = U.to_numpy().T
        dtype = float if all([np.isrealobj(x) for x in [U, *[op._arr for op in self._ops.ravel()]]]) else complex
        y = np.zeros((self.range.dim, U.shape[1]), dtype=dtype)
        m, n = self._ops.shape
        for i, j in np.ndindex(m, n):
            op = self._ops[i, j]
            a, b = op.source.dim, op.range.dim
            y[i::m][:b] += op.apply(op.source.from_numpy(U[j::n][:a].T), mu=mu).to_numpy().T
        return self.range.from_numpy(y.T)

    @property
    def H(self):
        adjoint_ops = np.zeros_like(self._ops).T
        for i, j in np.ndindex(*adjoint_ops.shape):
            adjoint_ops[i, j] = self._ops[j, i].H
        return self.with_(_ops=adjoint_ops, source_id=self.range_id, range_id=self.source_id,
                          name=self.name + '_adjoint')
