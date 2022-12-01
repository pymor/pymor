# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np
from scipy.linalg import solve, solve_triangular

from pymor.analyticalproblems.expressions import parse_expression
from pymor.core.base import abstractmethod
from pymor.core.config import config
from pymor.parameters.base import ParametricObject, Mu
from pymor.parameters.functionals import ParameterFunctional


class Function(ParametricObject):
    """Interface for |Parameter| dependent analytical functions.

    Every |Function| is a map of the form ::

       f(μ): Ω ⊆ R^d -> R^(shape_range)

    The returned values are |NumPy arrays| of arbitrary (but fixed)
    shape. Note that NumPy distinguishes between one-dimensional
    arrays of length 1 (with shape `(1,)`) and zero-dimensional
    scalar arrays (with shape `()`). In pyMOR, we usually
    expect scalar-valued functions to have `shape_range == ()`.

    While the function might raise an error if it is evaluated
    for an argument not in the domain Ω, the exact behavior is left
    undefined.

    Functions are vectorized in the sense, that if `x.ndim == k`, then ::

       f(x, μ)[i0, i1, ..., i(k-2)] == f(x[i0, i1, ..., i(k-2)], μ).

    In particular, `f(x, μ).shape == x.shape[:-1] + shape_range`.

    Attributes
    ----------
    dim_domain
        The dimension d > 0.
    shape_range
        The shape of the function values.
    """

    @abstractmethod
    def evaluate(self, x, mu=None):
        """Evaluate the function for given argument `x` and |parameter values| `mu`."""
        pass

    def __call__(self, x, mu=None):
        """Shorthand for :meth:`~Function.evaluate`."""
        return self.evaluate(x, mu)

    def _add_sub(self, other, sign):
        if isinstance(other, Number) and other == 0:
            return self
        elif not isinstance(other, Function):
            other = np.array(other)
            assert other.shape == self.shape_range
            if np.all(other == 0.):
                return self
            other = ConstantFunction(other, dim_domain=self.dim_domain)

        if self.name != 'LincombFunction' or not isinstance(self, LincombFunction):
            if other.name == 'LincombFunction' and isinstance(other, LincombFunction):
                functions = (self,) + other.functions
                coefficients = (1.,) + (other.coefficients if sign == 1. else tuple(-c for c in other.coefficients))
            else:
                functions, coefficients = (self, other), (1., sign)
        elif other.name == 'LincombFunction' and isinstance(other, LincombFunction):
            functions = self.functions + other.functions
            coefficients = self.coefficients + (other.coefficients if sign == 1.
                                                else tuple(-c for c in other.coefficients))
        else:
            functions, coefficients = self.functions + (other,), self.coefficients + (sign,)

        return LincombFunction(functions, coefficients)

    def _radd_sub(self, other, sign):
        assert not isinstance(other, Function)  # handled by __add__/__sub__
        if isinstance(other, Number) and other == 0:
            return self

        other = np.array(other)
        assert other.shape == self.shape_range
        if np.all(other == 0.):
            return self
        other = ConstantFunction(other, dim_domain=self.dim_domain)

        if self.name != 'LincombFunction' or not isinstance(self, LincombFunction):
            functions, coefficients = (other, self), (1., sign)
        else:
            functions = (other,) + self.functions
            coefficients = (1.,) + (self.coefficients if sign == 1. else tuple(-c for c in self.coefficients))

        return LincombFunction(functions, coefficients)

    def __add__(self, other):
        return self._add_sub(other, 1.)

    def __sub__(self, other):
        return self._add_sub(other, -1.)

    def __radd__(self, other):
        return self._radd_sub(other, 1.)

    def __rsub__(self, other):
        return self._radd_sub(other, -1.)

    def __mul__(self, other):
        if not isinstance(other, (Number, ParameterFunctional, Function)):
            return NotImplemented
        if isinstance(other, (Number, ParameterFunctional)):
            return LincombFunction([self], [other])
        if self.name != 'ProductFunction' or not isinstance(self, ProductFunction):
            if isinstance(other, ProductFunction) and other.name == 'ProductFunction':
                return other.with_(functions=other.functions + [self])
            else:
                return ProductFunction([self, other])
        elif isinstance(other, ProductFunction) and other.name == 'ProductFunction':
            functions = self.functions + other.functions
            return ProductFunction(functions)
        else:
            return self.with_(functions=self.functions + [other])

    __rmul__ = __mul__

    def __neg__(self):
        return LincombFunction([self], [-1.])


class ConstantFunction(Function):
    """A constant |Function|

    Defined as ::

        f: R^d -> R^shape(c), f(x) = c

    Parameters
    ----------
    value
        The constant c.
    dim_domain
        The dimension d.
    name
        The name of the function.
    """

    def __init__(self, value=np.array(1.0), dim_domain=1, name=None):
        assert dim_domain > 0
        assert isinstance(value, (Number, np.ndarray))
        value = np.array(value)
        self.__auto_init(locals())
        self.shape_range = value.shape

    def __str__(self):
        return f'{self.name}: x -> {self.value}'

    def evaluate(self, x, mu=None):
        x = np.array(x, copy=False, ndmin=1)
        assert x.shape[-1] == self.dim_domain
        if x.ndim == 1:
            return np.array(self.value)
        else:
            return np.tile(self.value, x.shape[:-1] + (1,) * len(self.shape_range))

    def to_fenics(self, mesh):
        """Convert to ufl expression over dolfin mesh.

        Parameters
        ----------
        mesh
            The dolfin mesh object.

        Returns
        -------
        coeffs
            |NumPy array| of shape `self.shape_range` where each entry is an ufl
            expression.
        params
            Dict mapping parameter names to lists of dolfin `Constants` which are
            used in the ufl expressions for the corresponding parameter values.
        """
        config.require('FENICS')
        from dolfin import Constant
        return np.vectorize(Constant)(self.value), {}


class GenericFunction(Function):
    """Wrapper making an arbitrary Python function between |NumPy arrays| a proper |Function|.

    Note that a :class:`GenericFunction` can only be :mod:`pickled <pymor.core.pickle>`
    if the function it is wrapping can be pickled (cf. :func:`~pymor.core.pickle.dumps_function`).
    For this reason, it is usually preferable to use :class:`ExpressionFunction`
    instead of :class:`GenericFunction`.

    Parameters
    ----------
    mapping
        The function to wrap. If `parameters` is `None`, the function is of
        the form `mapping(x)`. If `parameters` is not `None`, the function has
        to have the signature `mapping(x, mu)`. Moreover, the function is expected
        to be vectorized, i.e.::

            mapping(x).shape == x.shape[:-1] + shape_range.

    dim_domain
        The dimension of the domain.
    shape_range
        The shape of the values returned by the mapping.
    parameters
        The |Parameters| the mapping accepts.
    name
        The name of the function.
    """

    def __init__(self, mapping, dim_domain=1, shape_range=(), parameters={}, name=None):
        assert dim_domain > 0
        assert isinstance(shape_range, (Number, tuple))
        if not isinstance(shape_range, tuple):
            shape_range = (shape_range,)
        self.parameters_own = parameters
        self.__auto_init(locals())

    def __str__(self):
        return f'{self.name}: x -> {self.mapping}'

    def evaluate(self, x, mu=None):
        assert self.parameters.assert_compatible(mu)
        x = np.array(x, copy=False, ndmin=1)
        assert x.shape[-1] == self.dim_domain

        if self.parametric:
            v = self.mapping(x, mu=mu)
        else:
            v = self.mapping(x)

        if v.shape != x.shape[:-1] + self.shape_range:
            assert v.shape[:len(x.shape) - 1] == x.shape[:-1]
            v = v.reshape(x.shape[:-1] + self.shape_range)

        return v


class ExpressionFunction(GenericFunction):
    """Turns a Python expression given as a string into a |Function|.

    Some |NumPy| arithmetic functions like 'sin', 'log', 'min' are supported.
    For a full list see the `functions` class attribute.

    .. warning::
       :meth:`eval` is used to evaluate the given expression.
       Using this class with expression strings from untrusted sources will cause
       mayhem and destruction!

    Parameters
    ----------
    expression
        A Python expression of one variable and the `parameters`, given as a `str`.
    dim_domain
        The dimension of the domain.
    parameters
        The |Parameters| the expression accepts.
    values
        Dictionary of additional constants that can be used in `expression`
        with their corresponding value.
    variable
        Name of the input variable in the given expression.
    name
        The name of the function.
    """

    def __init__(self, expression, dim_domain=1, parameters={}, values={}, variable='x', name=None):
        params = parameters.copy()
        params[variable] = dim_domain
        self.expression_obj = parse_expression(expression, parameters=params, values=values)
        super().__init__(self.expression_obj.to_numpy([variable]),
                         dim_domain, self.expression_obj.shape, parameters, name)
        self.__auto_init(locals())

    def to_fenics(self, mesh):
        """Convert to ufl expression over dolfin mesh.

        Parameters
        ----------
        mesh
            The dolfin mesh object.

        Returns
        -------
        coeffs
            |NumPy array| of shape `self.shape_range` where each entry is an ufl
            expression.
        params
            Dict mapping parameter names to lists of dolfin `Constants` which are
            used in the ufl expressions for the corresponding parameter values.
        """
        return self.expression_obj.to_fenics(mesh)

    def __reduce__(self):
        return (ExpressionFunction,
                (self.expression, self.dim_domain, self.parameters, self.values, self.variable,
                 getattr(self, '_name', None)))

    def __str__(self):
        return f'{{x -> {self.expression}}}'


class LincombFunction(Function):
    """A |Function| representing a linear combination of |Functions|.

    The linear coefficients can be provided either as scalars or as
    |ParameterFunctionals|.

    Parameters
    ----------
    functions
        List of |Functions| whose linear combination is formed.
    coefficients
        A list of linear coefficients. A linear coefficient can
        either be a fixed number or a |ParameterFunctional|.
    name
        Name of the function.

    Attributes
    ----------
    functions
    coefficients
    """

    def __init__(self, functions, coefficients, name=None):
        assert len(functions) > 0
        assert len(functions) == len(coefficients)
        assert all(isinstance(f, Function) for f in functions)
        assert all(isinstance(c, (ParameterFunctional, Number)) for c in coefficients)
        assert all(f.dim_domain == functions[0].dim_domain for f in functions[1:])
        assert all(f.shape_range == functions[0].shape_range for f in functions[1:])
        functions = tuple(functions)
        coefficients = tuple(coefficients)

        self.__auto_init(locals())
        self.dim_domain = functions[0].dim_domain
        self.shape_range = functions[0].shape_range

    def evaluate_coefficients(self, mu):
        """Compute the linear coefficients for given |parameter values| `mu`."""
        assert self.parameters.assert_compatible(mu)
        return np.array([c.evaluate(mu) if hasattr(c, 'evaluate') else c for c in self.coefficients])

    def evaluate(self, x, mu=None):
        assert self.parameters.assert_compatible(mu)
        coeffs = self.evaluate_coefficients(mu)
        return sum(c * f(x, mu) for c, f in zip(coeffs, self.functions))


class ProductFunction(Function):
    """A |Function| representing a product of |Functions|.

    Parameters
    ----------
    functions
        List of |Functions| whose product is formed.
    name
        Name of the function.

    Attributes
    ----------
    functions
    """

    def __init__(self, functions, name=None):
        assert len(functions) > 0
        assert all(isinstance(f, Function) for f in functions)
        assert all(f.dim_domain == functions[0].dim_domain for f in functions[1:])
        assert all(f.shape_range == functions[0].shape_range for f in functions[1:])
        self.__auto_init(locals())
        self.dim_domain = functions[0].dim_domain
        self.shape_range = functions[0].shape_range

    def evaluate(self, x, mu=None):
        assert self.parameters.assert_compatible(mu)
        return np.prod([f(x, mu) for f in self.functions], axis=0)


class BitmapFunction(Function):
    """Define a 2D |Function| via a grayscale image.

    Parameters
    ----------
    filename
        Path of the image representing the function.
    bounding_box
        Lower left and upper right coordinates of the domain of the function.
    range
        A pixel of value p is mapped to `(p / 255.) * range[1] + range[0]`.
    """

    dim_domain = 2
    shape_range = ()

    def __init__(self, filename, bounding_box=None, range=None):
        bounding_box = bounding_box or [[0., 0.], [1., 1.]]
        range = range or [0., 1.]
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("PIL is needed for loading images. Try 'pip install pillow'") from e
        img = Image.open(filename)
        if not img.mode == "L":
            self.logger.warning("Image " + filename + " not in grayscale mode. Converting to grayscale.")
            img = img.convert('L')
        self.__auto_init(locals())
        self.bitmap = np.array(img).T[:, ::-1]
        self.lower_left = np.array(bounding_box[0])
        self.size = np.array(bounding_box[1] - self.lower_left)

    def evaluate(self, x, mu=None):
        indices = np.maximum(np.floor((x - self.lower_left) * np.array(self.bitmap.shape) / self.size).astype(int), 0)
        F = (self.bitmap[np.minimum(indices[..., 0], self.bitmap.shape[0] - 1),
                         np.minimum(indices[..., 1], self.bitmap.shape[1] - 1)]
             * ((self.range[1] - self.range[0]) / 255.)
             + self.range[0])
        return F


class EmpiricalInterpolatedFunction(LincombFunction):
    """Empirically interpolated |Function|.

    Instantiated by :func:`~pymor.algorithm.ei.interpolate_function`.

    Parameters
    ----------
    function
        The |Function| to interpolate.
    interpolation_points
        |NumPy array| containing the coordinates at which the function
        is interpolated. Typically `X[dofs]` where `X` is the array of
        evaluation points used for snapshot data generation and `dofs`
        is returned by :func:`~pymor.algorithm.ei.ei_greedy`.
    interpolation_matrix
        The interpolation matrix corresponding to the selected interpolation
        basis vectors and `interpolation_points` as returned by
        :func:`pymor.algorithms.ei.ei_greedy`.
    triangular
        Whether or not `interpolation_matrix` is lower triangular with unit
        diagonal.
    snapshot_mus
        List of |parameter values| for which the snapshot data for
        :func:`pymor.algorithms.ei.ei_greedy` has been computed.
    snapshot_coefficients
        Matrix of linear coefficients s.t. the i-th interpolation basis vector
        is given by a linear combination of the functions corresponding to
        `snapshot_mus` with the i-th row of `snapshot_coefficients` as
        coefficients. Returned by :func:`~pymor.algorithm.ei.ei_greedy` as
        `data['coefficients']`.
    evaluation_points
        Optional |NumPy array| of coordinates at which the function has been
        evaluated to obtain the snapshot data. If the same evaluation points
        are used to evaluate :class:`EmpiricalInterpolatedFunction`, then
        re-evaluation of the snapshot data at the evaluation points can be
        avoided, when this argument is specified together with `basis_evaluations`.
    basis_evaluations
        Optional |Numpy array| of evaluations of the interpolation basis at
        `evaluation_points`. Corresponds to the `basis` return value of
        :func:`~pymor.algorithms.ei.ei_greedy`.
    """

    def __init__(self, function, interpolation_points, interpolation_matrix, triangular,
                 snapshot_mus, snapshot_coefficients,
                 evaluation_points=None, basis_evaluations=None,
                 name=None):

        assert isinstance(function, Function)
        if len(function.shape_range) > 0:
            raise NotImplementedError
        assert isinstance(interpolation_points, np.ndarray) and interpolation_points.ndim == 2 and \
            interpolation_points.shape[1] == function.dim_domain
        assert isinstance(interpolation_matrix, np.ndarray) and \
            interpolation_matrix.shape == (len(interpolation_points),) * 2
        assert all(isinstance(mu, Mu) for mu in snapshot_mus)
        assert isinstance(snapshot_coefficients, np.ndarray) and \
            snapshot_coefficients.shape == (len(interpolation_points), len(snapshot_mus))
        assert (evaluation_points is None) == (basis_evaluations is None)
        assert evaluation_points is None or isinstance(evaluation_points, np.ndarray) and \
            evaluation_points.ndim == 2 and evaluation_points.shape[1] == function.dim_domain
        assert basis_evaluations is None or isinstance(basis_evaluations, np.ndarray) and \
            basis_evaluations.shape == (len(interpolation_points), len(evaluation_points)) + function.shape_range

        self.__auto_init(locals())
        functions = [EmpiricalInterpolatedFunctionBasisFunction(self, i) for i in range(len(interpolation_points))]
        coefficients = [EmpiricalInterpolatedFunctionFunctional(self, i) for i in range(len(interpolation_points))]
        super().__init__(functions, coefficients)
        self._last_evaluation_points, self._last_basis_evaluations = evaluation_points, basis_evaluations
        if self._last_evaluation_points is not None:
            self._last_evaluation_points.flags.writeable = False
            self._last_basis_evaluations.flags.writeable = False
        self._last_mu = 'NONE'

    def _update_coefficients(self, mu):
        if mu is self._last_mu:
            return
        assert self.parameters.assert_compatible(mu)
        self._last_mu = mu
        fx = self.function.evaluate(self.interpolation_points, mu=mu)
        if self.triangular:
            self._last_interpolation_coefficients = solve_triangular(
                self.interpolation_matrix, fx, lower=True, unit_diagonal=True
            )
        else:
            self._last_interpolation_coefficients = solve(self.interpolation_matrix, fx)

    def _update_evaluation_points(self, x):
        if self._last_evaluation_points is not None and np.all(x == self._last_evaluation_points):
            return
        if self._last_evaluation_points is not None:
            self.logger.info('Evaluating function snapshots for new evaluation points.')
        x.flags.writeable = False
        self._last_evaluation_points = x
        snapshot_evaluations = np.array([self.function.evaluate(x, mu=mu) for mu in self.snapshot_mus])
        self._last_basis_evaluations = self.snapshot_coefficients @ snapshot_evaluations.T


class EmpiricalInterpolatedFunctionFunctional(ParameterFunctional):

    def __init__(self, interpolated_function, index):
        self.__auto_init(locals())
        # we explicitly have to set the parameters since interpolation_points isn't initialized yet
        self.parameters = interpolated_function.function.parameters

    def evaluate(self, mu=None):
        self.interpolated_function._update_coefficients(mu)
        return self.interpolated_function._last_interpolation_coefficients[self.index]


class EmpiricalInterpolatedFunctionBasisFunction(Function):

    def __init__(self, interpolated_function, index):
        self.__auto_init(locals())
        self.dim_domain = interpolated_function.function.dim_domain
        self.shape_range = interpolated_function.function.shape_range
        self.parameters = {}

    def evaluate(self, x, mu=None):
        self.interpolated_function._update_evaluation_points(x)
        return self.interpolated_function._last_basis_evaluations[self.index].copy()
