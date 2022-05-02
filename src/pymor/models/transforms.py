import numpy as np
from scipy.linalg import null_space

from pymor.core.base import ImmutableObject


class MoebiusTransformation(ImmutableObject):
    r"""Maps the Riemann sphere onto itself.

    A Moebius transformation

    .. math::
        M(s) = \frac{as+b}{cs+b}

    is determined by the coefficients :math:`a,b,c,d\in\mathbb{C}`. The Moebius transformations form
    a group under composition, therefore the `__matmul__` operator is defined to yield a
    |MoebiusTransformation| if both factors are |MoebiusTransformation|s.

    Parameters
    ----------
    coefficients
        A tuple, list or |NumPy array| containing the four coefficients `a,b,c,d`.
    normalize
        If `True`, the coefficients are normalized, i.e. :math:`ad-bc=1`. Defaults to `False`.
    name
        Name of the transformation.
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
        """Constructs a Moebius transformation from three points and their images.

        A Moebius transformation is completely determined by the images of three distinct points on
        the Riemann sphere under transformation.

        Parameters
        ----------
        w
            A tuple, list or |NumPy array| of three complex numbers that are transformed.
        z
            A tuple, list or |NumPy array| of three complex numbers represent the images of `w`.
            Defaults to `(0, 1, np.inf)`.
        name
            Name of the transformation.
        """
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
        """Returns the inverse Moebius transformation by applying the inversion formula.

        Parameters
        ----------
        normalize
            If `True`, the coefficients are normalized, i.e. :math:`ad-bc=1`. Defaults to `False`.
        """
        a, b, c, d = self.coefficients
        coefficients = np.array([d, -b, -c, a])
        return MoebiusTransformation(coefficients, normalize=normalize, name=self.name + '_inverse')

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
        assert isinstance(other, MoebiusTransformation)
        M = self.coefficients.reshape(2, 2) @ other.coefficients.reshape(2, 2)
        return MoebiusTransformation(M.ravel())

    def __str__(self):
        return (
            f'{self.name}: C^1 --> C^1\n'
            f'       ({self.coefficients[0]:.1f})*z + ({self.coefficients[1]:.1f})\n'
            'f(z) = --------------------------------\n'
            f'       ({self.coefficients[2]:.1f})*z + ({self.coefficients[3]:.1f})'
        )


class BilinearTransform(MoebiusTransformation):
    r"""The bilinear transform also known as Tustin's method.

    The bilinear transform can be seen as the first order approximation of the natural logarithm
    that maps the z-plane onto the s-plane. The approximation is given by

    .. math::
        z = \frac{1+xs}{1-xs},

    where `x` is an arbitrary number. Usually, this is chosen as the step size of the numerical
    integration trapezoid rule, i.e. the sampling time of a discrete-time |LTIModel|.

    Parameters
    ----------
    x
        An arbitrary number that defines the |BilinearTransform|.
    name
        Name of the transform.
    """

    def __init__(self, x, name=None):
        assert x > 0
        super().__init__([x, -x, 1, 1], name=name)
        self.__auto_init(locals())


class CayleyTransform(MoebiusTransformation):
    r"""Maps the upper complex half-plane to the unit disk.

    The Cayley transform is defined as

    .. math::
        f(s) = \frac{s-i}{s+i}.

    Parameters
    ----------
    name
        Name of the transform.
    """

    def __init__(self, name=None):
        super().__init__([1, -1j, 1, 1j], name=name)
        self.__auto_init(locals())
