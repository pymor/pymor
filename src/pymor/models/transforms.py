import numpy as np
from scipy.linalg import null_space

from pymor.core.base import ImmutableObject


class MoebiusTransform(ImmutableObject):
    """A Moebius transform operator.

    Maps complex numbers to complex numbers.
    The transform coefficients can be normalized.
    Contains a method for constructing an inverse mapping, as well as a constructor that takes three
    points and their images. The Moebius transforms form a group under composition, so __matmul__ is
    defined to yield Moebius Transforms if both factors are Moebius Transforms.
    """

    def __init__(self, coefficients, normalize=False, name=None):
        assert len(coefficients) == 4
        coefficients = np.array(coefficients, dtype=complex)
        assert coefficients[0]*coefficients[3] != coefficients[1]*coefficients[2]

        if normalize:
            coefficients /= np.sqrt(coefficients[0]*coefficients[3]-coefficients[2]*coefficients[3])

        self.__auto_init(locals())

    @classmethod
    def from_points(cls, w, z=(0, 1, np.inf), name=None):
        assert len(z) == 3
        assert len(w) == 3

        A = np.zeros([3, 4])
        for i in range(3):
            if np.isinf(w[i]) and np.isinf(z[i]):
                A[i] = [0, 0, -1, 0]
            elif np.isinf(w[i]):
                A[i] = [0, 0, -z[i], -1]
            elif np.isinf(z[i]):
                A[i] = [1, 0, -w[i], 0]
            else:
                A[i] = [z[i], 1, -w[i]*z[i], -w[i]]

        return cls(null_space(A).ravel(), name=name)

    def inverse(self, normalize=False):
        a, b, c, d = self.coefficients
        coefficients = np.array([d, -b, -c, a])
        return MoebiusTransform(coefficients, normalize=normalize, name=self.name + '_inverse')

    def _mapping(self, x):
        a, b, c, d = self.coefficients
        if c != 0 and np.allclose(x, -d / c):
            return np.inf
        elif c != 0 and np.isinf(x):
            return a / c
        elif c == 0 and np.isinf(x):
            return np.inf
        else:
            return (a * x + b) / (c * x + d)

    def __call__(self, x):
        x = np.squeeze(np.array(x, dtype=complex))
        assert x.ndim <= 1
        return np.vectorize(self._mapping)(x)

    def __matmul__(self, other):
        assert isinstance(other, MoebiusTransform)
        M = self.coefficients.reshape(2, 2) @ other.coefficients.reshape(2, 2)
        return MoebiusTransform(M.ravel())

    def __str__(self):
        return (
            f'{self.name}: C^1 --> C^1\n'
            f'       ({self.coefficients[0]:.1f})*z + ({self.coefficients[1]:.1f})\n'
            'f(z) = --------------------------------\n'
            f'       ({self.coefficients[2]:.1f})*z + ({self.coefficients[3]:.1f})'
        )


class BilinearTransform(MoebiusTransform):
    def __init__(self, x, normalize=False, name=None):
        assert x > 0
        super().__init__([x, -x, 1, 1], normalize=normalize, name=name)
        self.__auto_init(locals())


class CayleyTransform(MoebiusTransform):
    def __init__(self, normalize=False, name=None):
        super().__init__([1, -1j, 1, 1j], normalize=normalize, name=name)
        self.__auto_init(locals())
