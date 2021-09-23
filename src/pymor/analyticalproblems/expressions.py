# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module contains a basic symbolic expression library.

The library is used by |ExpressionFunction| and |ExpressionParameterFunctional|
by calling :func:`parse_expression`, which parses `str` expressions by replacing
the names in the string with objects from this module. The result is an
:class:`Expression` object, which can be converted to a |NumPy|-vectorized function
using :meth:`Expression.to_numpy`. In the future, more conversion routines will
be added to make the same :class:`Expression` usable for :mod:`pymor.discretizers`
that use external PDE solvers. Further advantages of using this expression library:

- meaningful error messages are generated at parse time of the `str` expression,
  instead of hard-to-debug errors in lambda functions at evaluation time,
- expressions are automatically correctly vectorized. In particular, there is no
  longer a need to add `...` to indexing expressions,
- the shape of the resulting expressions is automatically determined.

In the future, we will also provide support for symbolic differentiation of the
given :class:`Expressions`.

Every :class:`Expression` is built from the following atoms:

- a :class:`Constant`, which is a fixed value of arbitrary shape,
- a :class:`Parameter`, which is a variable of a fixed one-dimensional shape.

Note that both |Parameters| and input variables are treated as a :class:`Parameter`
in the expression. Only when calling, e.g., :meth:`~Expression.to_numpy`, it is
determined which :class:`Parameter` belongs to the resulting function's |Parameters|
and which :class:`Parameter` is treated as an input variable.

More complex expressions can be built using:

- :class:`binary operators <BinaryOp>`,
- :class:`negation <Neg>`,
- :class:`function calls <UnaryFunctionCall>`,
- :class:`indexing <Indexed>`,
- :class:`array construction <Array>`.

For binary operations of :class:`Expressions <Expression>` of different shape,
the usual broadcasting rules apply.
"""


from numbers import Number
from itertools import zip_longest
import numpy as np

from pymor.parameters.base import ParametricObject


builtin_max = max


def parse_expression(expression, parameters={}, values={}):
    if isinstance(expression, Expression):
        return expression
    locals_dict = {name: Parameter(name, dim) for name, dim in parameters.items()}
    return _convert_to_expression(eval(expression, dict(globals(), **values), locals_dict))


class Expression(ParametricObject):
    """A symbolic math expression

    Attributes
    ----------
    shape
        The shape of the object this expression evaluates to
        in the sense of |NumPy|.
    """

    shape = None

    def to_numpy(self, variables):
        """Convert expression to a |NumPy|-vectorized callable.

        Parameters
        ----------
        variables
            List of names of :class:`~Parameters <Parameter>` in
            the expression which shall be treated as input variables.
        """
        expression = self.numpy_expr()
        code = compile(expression, '<expression>', 'eval')

        def wrapper(*args, mu={}):
            if not variables and args:
                assert len(args) == 1
                mu = args[0]
                args = []
            assert all(_broadcastable_shapes(args[0].shape[:-1], a.shape[:-1]) for a in args[1:])
            if len(args) == 0:
                input_shape = ()
            elif len(args) == 1:
                input_shape = args[0].shape[:-1]
            else:
                input_shape = (tuple(builtin_max(*s)
                                     for s in zip_longest(*(a.shape[-2::-1] for a in args),
                                                          fillvalue=1)))[::-1]
            all_args = dict(mu) if mu else {}
            all_args.update({k: v for k, v in zip(variables, args)})
            result = np.broadcast_to(eval(code,
                                          _numpy_functions,
                                          all_args),
                                     input_shape + self.shape)
            return result

        return wrapper

    def numpy_expr(self):
        """Called by :meth:`~Expression.to_numpy`."""
        raise NotImplementedError

    def __getitem__(self, index):
        return Indexed(self, index)

    def __add__(self, other):
        return Sum(self, _convert_to_expression(other))

    def __radd__(self, other):
        return Sum(_convert_to_expression(other), self)

    def __sub__(self, other):
        return Diff(self, _convert_to_expression(other))

    def __rsub__(self, other):
        return Diff(_convert_to_expression(other), self)

    def __mul__(self, other):
        return Prod(self, _convert_to_expression(other))

    def __rmul__(self, other):
        return Prod(_convert_to_expression(other), self)

    def __truediv__(self, other):
        return Div(self, _convert_to_expression(other))

    def __rtruediv__(self, other):
        return Div(_convert_to_expression(other), self)

    def __pow__(self, other):
        return Pow(self, _convert_to_expression(other))

    def __rpow__(self, other):
        return Pow(_convert_to_expression(other), self)

    def __mod__(self, other):
        return Mod(self, _convert_to_expression(other))

    def __rmod__(self, other):
        return Mod(_convert_to_expression(other), self)

    def __neg__(self):
        return Neg(self)

    def __le__(self, other):
        return LE(self, _convert_to_expression(other))

    def __ge__(self, other):
        return GE(self, _convert_to_expression(other))

    def __lt__(self, other):
        return LT(self, _convert_to_expression(other))

    def __gt__(self, other):
        return GT(self, _convert_to_expression(other))


class BaseConstant(Expression):
    """A constant value."""

    numpy_symbol = None
    shape = ()

    def numpy_expr(self):
        return f'array({self.numpy_symbol}, ndmin={len(self.shape)}, copy=False)'

    def __str__(self):
        return str(self.numpy_symbol)


class Constant(BaseConstant):
    """A constant value given by a |NumPy| array."""

    def __init__(self, value):
        self.value = value = np.array(value)
        self.numpy_symbol = repr(value.tolist())
        self.shape = value.shape

    def __str__(self):
        return str(self.value)


class Parameter(Expression):
    """A free parameter in an :class:`Expression`.

    Parameters represent both pyMOR |Parameters| as well as
    input variables. Each parameter is a vector of shape `(dim,)`.

    Parameters
    ----------
    name
        The name of the parameter.
    dim
        The dimension of the parameter.
    """

    def __init__(self, name, dim):
        self.name, self.dim = name, dim
        self.numpy_symbol = name
        self.shape = (dim,)
        self.parameters_own = {name: dim}

    def numpy_expr(self):
        return str(self.numpy_symbol)

    def __str__(self):
        return str(self.numpy_symbol)


class Array(Expression):
    """An array of scalar-valued :class:`Expressions <Expression>`."""

    def __init__(self, array):
        array = np.array(array)
        for i, v in np.ndenumerate(array):
            if not isinstance(v, Expression):
                raise ValueError(f'Array entry {v} at index {i} is not an Expression.')
            if v.shape not in ((), (1,)):
                raise ValueError(f'Array entry {v} at index {i} is not scalar valued (shape: {v.shape}).')
        self.array = np.vectorize(lambda x: x[0] if x.shape else x)(array)
        self.shape = array.shape

    def numpy_expr(self):
        entries = [v.numpy_expr() for v in self.array.flat]
        return f'(lambda a: array(a).T.reshape(a[0].shape + {self.shape}))(broadcast_arrays({", ".join(entries)}))'

    def __str__(self):
        expr_array = np.vectorize(str)(self.array)
        return str(expr_array)


class BinaryOp(Expression):
    """Compound :class:`Expression` of a binary operator acting on two sub-expressions."""

    numpy_symbol = None

    def __init__(self, first, second):
        if not _broadcastable_shapes(first.shape, second.shape):
            raise ValueError(f'Incompatible shapes of expressions "{first}" and "{second}" with shapes '
                             f'{first.shape} and {second.shape} for binary operator {self.numpy_symbol}')
        self.first, self.second = first, second
        self.shape = tuple(builtin_max(f, s)
                           for f, s in zip_longest(first.shape[::-1], second.shape[::-1], fillvalue=1))[::-1]

    def numpy_expr(self):
        first_ind = second_ind = ''
        if len(self.first.shape) > len(self.second.shape):
            second_ind = (['...']
                          + ['newaxis'] * (len(self.first.shape) - len(self.second.shape))
                          + [':'] * len(self.second.shape))
            second_ind = '[' + ','.join(second_ind) + ']'
        if len(self.first.shape) < len(self.second.shape):
            first_ind = (['...']
                         + ['newaxis'] * (len(self.second.shape) - len(self.first.shape))
                         + [':'] * len(self.first.shape))
            first_ind = '[' + ','.join(first_ind) + ']'

        return f'({self.first.numpy_expr()}{first_ind} {self.numpy_symbol} {self.second.numpy_expr()}{second_ind})'

    def __str__(self):
        return f'({self.first} {self.numpy_symbol} {self.second})'


class Neg(Expression):
    """Negated :class:`Expression`."""

    def __init__(self, operand):
        self.operand = operand
        self.shape = operand.shape

    def numpy_expr(self):
        return f'(- {self.operand.numpy_expr()})'

    def __str__(self):
        return f'(- {self.operand})'


class Indexed(Expression):
    """Indexed :class:`Expression`."""

    def __init__(self, base, index):
        if not isinstance(index, int) and \
                not (isinstance(index, tuple) and all(isinstance(i, int) for i in index)):
            raise ValueError(f'Indices must be ints or tuples of ints (given: {index})')
        if isinstance(index, int):
            index = (index,)
        if not len(index) == len(base.shape):
            raise ValueError(f'Wrong number of indices (given: {index} for expression "{base}" of shape {base.shape})')
        if not all(0 <= i < s for i, s in zip(index, base.shape)):
            raise ValueError(f'Invalid index (given: {index} for expression "{base}" of shape {base.shape})')
        self.base, self.index = base, index
        self.shape = base.shape[len(index):]

    def numpy_expr(self):
        index = ['...'] + [repr(i) for i in self.index]
        return f'{self.base.numpy_expr()}[{",".join(index)}]'

    def __str__(self):
        index = [str(i) for i in self.index]
        return f'{self.base}[{",".join(index)}]'


class UnaryFunctionCall(Expression):
    """Compound :class:`Expression` of an unary function applied to a sub-expression.

    The function is applied component-wise.
    """

    numpy_symbol = None

    def __init__(self, *arg):
        if len(arg) != 1:
            raise ValueError(f'{self.numpy_symbol} takes a single argument (given: {arg})')
        self.arg = _convert_to_expression(arg[0])
        self.shape = self.arg.shape

    def numpy_expr(self):
        return f'{self.numpy_symbol}({self.arg.numpy_expr()})'

    def __str__(self):
        return f'{self.numpy_symbol}({self.arg})'


class UnaryReductionCall(Expression):
    """Compound :class:`Expression` of an unary function applied to a sub-expression.

    The function is applied to the entire vector/matrix/tensor the sub-expression evaluates to,
    returning a single number.
    """

    def __init__(self, *arg):
        if len(arg) != 1:
            raise ValueError(f'{self.numpy_symbol} takes a single argument (given {arg})')
        self.arg = _convert_to_expression(arg[0])
        self.shape = ()

    def numpy_expr(self):
        return (f'(lambda _a: {self.numpy_symbol}(_a.reshape(_a.shape[:-{len(self.arg.shape)}] + (-1,)), '
                f'axis=-1))({self.arg.numpy_expr()})')

    def __str__(self):
        return f'{self.numpy_symbol}({self.arg})'


def _convert_to_expression(obj):
    """Used internally to convert literals/list constructors to an :class:`Expression`."""
    if isinstance(obj, Expression):
        return obj
    if isinstance(obj, Number):
        return Constant(obj)
    obj = np.array(obj)
    if obj.dtype == object:
        if obj.shape:
            obj = np.vectorize(_convert_to_expression)(obj)
            return Array(obj)
        else:
            raise ValueError(f'Cannot convert {obj} to expression')
    else:
        return Constant(obj)


def _broadcastable_shapes(first, second):
    return all(f == s or f == 1 or s == 1 for f, s in zip(first[::-1], second[::-1]))


class Sum(BinaryOp):  numpy_symbol = '+'   # NOQA
class Diff(BinaryOp): numpy_symbol = '-'   # NOQA
class Prod(BinaryOp): numpy_symbol = '*'   # NOQA
class Div(BinaryOp):  numpy_symbol = '/'   # NOQA
class Pow(BinaryOp):  numpy_symbol = '**'  # NOQA
class Mod(BinaryOp):  numpy_symbol = '%'   # NOQA
class LE(BinaryOp):   numpy_symbol = '<='  # NOQA
class GE(BinaryOp):   numpy_symbol = '>='  # NOQA
class LT(BinaryOp):   numpy_symbol = '<'   # NOQA
class GT(BinaryOp):   numpy_symbol = '>'   # NOQA


class sin(UnaryFunctionCall):      numpy_symbol = 'sin'      # NOQA
class cos(UnaryFunctionCall):      numpy_symbol = 'cos'      # NOQA
class tan(UnaryFunctionCall):      numpy_symbol = 'tan'      # NOQA
class arcsin(UnaryFunctionCall):   numpy_symbol = 'arcsin'   # NOQA
class arccos(UnaryFunctionCall):   numpy_symbol = 'arccos'   # NOQA
class arctan(UnaryFunctionCall):   numpy_symbol = 'arctan'   # NOQA
class sinh(UnaryFunctionCall):     numpy_symbol = 'sinh'     # NOQA
class cosh(UnaryFunctionCall):     numpy_symbol = 'cosh'     # NOQA
class tanh(UnaryFunctionCall):     numpy_symbol = 'tanh'     # NOQA
class arcsinh(UnaryFunctionCall):  numpy_symbol = 'arcsinh'  # NOQA
class arccosh(UnaryFunctionCall):  numpy_symbol = 'arccosh'  # NOQA
class arctanh(UnaryFunctionCall):  numpy_symbol = 'arctanh'  # NOQA
class exp(UnaryFunctionCall):      numpy_symbol = 'exp'      # NOQA
class exp2(UnaryFunctionCall):     numpy_symbol = 'exp2'     # NOQA
class log(UnaryFunctionCall):      numpy_symbol = 'log'      # NOQA
class log2(UnaryFunctionCall):     numpy_symbol = 'log2'     # NOQA
class log10(UnaryFunctionCall):    numpy_symbol = 'log10'    # NOQA
class sqrt(UnaryFunctionCall):     numpy_symbol = 'sqrt'     # NOQA
class abs(UnaryFunctionCall):      numpy_symbol = 'abs'      # NOQA
class sign(UnaryFunctionCall):     numpy_symbol = 'sign'     # NOQA


class angle(UnaryFunctionCall):

    numpy_symbol = 'angle'

    def __init__(self, arg):
        if arg.shape[-1] != 2:
            raise ValueError
        self.arg = arg
        self.shape = arg.shape[:-1]


class norm(UnaryReductionCall): numpy_symbol = 'norm'    # NOQA
class min(UnaryReductionCall):  numpy_symbol = 'min'     # NOQA
class max(UnaryReductionCall):  numpy_symbol = 'max'     # NOQA
class sum(UnaryReductionCall):  numpy_symbol = 'sum'     # NOQA
class prod(UnaryReductionCall): numpy_symbol = 'prod'    # NOQA


class Pi(BaseConstant): numpy_symbol = 'pi'  # NOQA
class E(BaseConstant):  numpy_symbol = 'e'   # NOQA


pi = Pi()
e  = E()


_numpy_functions = {k: getattr(np, k) for k in {'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2',
                                                'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
                                                'exp', 'exp2', 'log', 'log2', 'log10', 'sqrt', 'abs', 'sign',
                                                'min', 'max', 'sum', 'prod',
                                                'pi', 'e',
                                                'array', 'broadcast_arrays', 'newaxis'}}

_numpy_functions['norm']  = np.linalg.norm
_numpy_functions['angle'] = lambda x: np.arctan2(x[..., 1], x[..., 0]) % (2*np.pi)  # np.angle uses different convention
