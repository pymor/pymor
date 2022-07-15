# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

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


import ast
from numbers import Number
import operator
from itertools import zip_longest
import numpy as np

from pymor.parameters.base import ParametricObject
from pymor.core.config import config


builtin_max = max


def parse_expression(expression, parameters={}, values={}):
    if isinstance(expression, Expression):
        return expression

    # convert parameters to Expressions
    locals_dict = {name: Parameter(name, dim) for name, dim in parameters.items()}

    # convert values to Expressions
    def parse_value(val):
        val = np.vectorize(lambda x: x if isinstance(x, Expression) else Constant(x))(val)
        return Array(val) if val.shape else val.item()

    values = {name: parse_value(val) for name, val in values.items()}

    # parse Expression
    tree = ast.parse(expression, mode='eval')

    # check if all names in given Expression are valid
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for name in names:
        if not ((name in globals() and isinstance(globals()[name], type) and issubclass(globals()[name], Expression))
                or (name in globals() and isinstance(globals()[name], Expression))
                or name in parameters
                or name in values):
            raise ValueError(f'Unknown name "{name}" in expression "{expression}"')

    # wrap all literals as Expressions
    transformed_tree = TransformLiterals().visit(tree)
    ast.fix_missing_locations(transformed_tree)

    # evaluate expression
    code = compile(transformed_tree, 'expression', mode='eval')
    try:
        expression = eval(code, dict(globals(), **values), locals_dict)
    except ValueError as e:
        raise ValueError(f'''
While parsing expression
\t{expression}
with parameters {parameters} and values {values} the following error occured:
\t{e}
''') from e

    if not isinstance(expression, Expression):
        raise ValueError(f'''
Malformed expression
\t{expression}
evaluates to {type(expression).__name__} instead of Expression object.
''')

    return expression


class TransformLiterals(ast.NodeTransformer):

    in_subscript = False

    def visit_Constant(self, node):
        if self.in_subscript:
            return node
        return ast.Call(ast.Name('Constant', ast.Load()),
                        [self.generic_visit(node)], [])

    # Python versions prior to 3.8 use ast.Num instead of ast.Constant
    def visit_Num(self, node):
        return self.visit_Constant(node)

    def visit_Subscript(self, node):
        base = self.visit(node.value)
        self.in_subscript = True
        slice = self.visit(node.slice)
        self.in_subscript = False
        return ast.Subscript(base, slice, node.ctx)

    def visit_List(self, node):
        return ast.Call(ast.Name('Array', ast.Load()),
                        [self.generic_visit(node)], [])


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

    def to_fenics(self, mesh, variable='x'):
        # sanity check for dolfin before running code
        config.require('FENICS')
        from dolfin import Constant
        from ufl import SpatialCoordinate
        assert variable not in self.parameters or self.parameters[variable] == mesh.topology().dim()
        params = {p: SpatialCoordinate(mesh) if p == variable else [Constant(0.) for _ in range(dim)]
                  for p, dim in self.parameters.items()}
        f_expr = self.fenics_expr(params)
        params.pop(variable, None)
        return f_expr, params

    def numpy_expr(self):
        """Called by :meth:`~Expression.to_numpy`."""
        raise NotImplementedError

    def fenics_expr(self, params):
        """Called by :meth:`~Expression.to_fenics`."""
        raise NotImplementedError

    def __getitem__(self, index):
        return Indexed(self, index)

    def __add__(self, other):
        return Sum(self, other)

    def __radd__(self, other):
        return Sum(other, self)

    def __sub__(self, other):
        return Diff(self, other)

    def __rsub__(self, other):
        return Diff(other, self)

    def __mul__(self, other):
        return Prod(self, other)

    def __rmul__(self, other):
        return Prod(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __pow__(self, other):
        return Pow(self, other)

    def __rpow__(self, other):
        return Pow(other, self)

    def __mod__(self, other):
        return Mod(self, other)

    def __rmod__(self, other):
        return Mod(other, self)

    def __neg__(self):
        return Neg(self)

    def __le__(self, other):
        return LE(self, other)

    def __ge__(self, other):
        return GE(self, other)

    def __lt__(self, other):
        return LT(self, other)

    def __gt__(self, other):
        return GT(self, other)

    def __bool__(self):
        raise TypeError("Cannot convert Expression to bool. (Don't use boolean operators or two-sided comparisons.)")


class BaseConstant(Expression):
    """A constant value."""

    numpy_symbol = None
    fenics_symbol = None
    shape = ()

    def numpy_expr(self):
        return f'array({self.numpy_symbol}, ndmin={len(self.shape)}, copy=False)'

    def fenics_expr(self, params):
        import ufl
        if self.fenics_symbol is None:
            raise NotImplementedError('No FEniCS symbol was given!')

        fenics_const = getattr(ufl, self.fenics_symbol)
        return np.array(fenics_const)

    def __str__(self):
        return str(self.numpy_symbol)


class Constant(BaseConstant):
    """A constant value given by a |NumPy| array."""

    shape = ()

    def __init__(self, value):
        if isinstance(value, np.ndarray):
            value = value.item()
        if not isinstance(value, Number):
            raise ValueError(f'Invalid Constant "{value}" of type {type(value).__name__} (expected Number).')
        self.value = value
        self.numpy_symbol = repr(value)

    def fenics_expr(self, params):
        from dolfin import Constant
        return np.array(Constant(self.value))

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
        if not isinstance(name, str):
            raise ValueError(f'Invalid name "{name}" for Parameter (must by string given {type(name).__name__}).')
        if not isinstance(dim, int):
            raise ValueError(f'Invalid dimension "{dim}" for Parameter {name} '
                             f'(must by int given {type(dim).__name__}).')
        self.name, self.dim = name, dim
        self.shape = (dim,)
        self.parameters_own = {name: dim}

    def numpy_expr(self):
        return str(self.name)

    def fenics_expr(self, params):
        return np.array(params[self.name])

    def __str__(self):
        return str(self.name)


class Array(Expression):
    """An array of scalar-valued :class:`Expressions <Expression>`."""

    def __init__(self, array):
        array = [a.array if isinstance(a, Array) else a for a in array]
        array = [a.tolist() if isinstance(a, np.ndarray) else a for a in array]
        A = np.array(array)
        for i, v in np.ndenumerate(A):
            if isinstance(v, (np.ndarray, list)):
                raise ValueError(f'Malformed Array construction {array} does not give ndarray of Expressions.')
            if not isinstance(v, Expression):
                raise ValueError(f'Entry "{v}" at index {i} of Array {array} is not an Expression '
                                 f'(type: {type(v).__name__}).')
            if v.shape != ():
                raise ValueError(f'Entry "{v}" at index {i} of Array {array} is not scalar valued (shape: {v.shape}).')
        self.array = A
        self.shape = A.shape

    def numpy_expr(self):
        entries = [v.numpy_expr() for v in self.array.flat]
        return (f'(lambda a: moveaxis(array(a), 0, -1).reshape(a[0].shape + {self.shape}))'
                f'(broadcast_arrays({", ".join(entries)}))')

    def _format_repr(self, max_width, verbosity):
        return super()._format_repr(max_width, verbosity, override={'array': repr(self.array.tolist())})

    def fenics_expr(self, params):
        return np.vectorize(lambda x: x.fenics_expr(params).item())(self.array)

    def __str__(self):

        def to_str(a):
            if not isinstance(a, np.ndarray):
                return str(a)
            return '[' + ', '.join(to_str(aa) for aa in a) + ']'

        return to_str(self.array)


class BinaryOp(Expression):
    """Compound :class:`Expression` of a binary operator acting on two sub-expressions."""

    numpy_symbol = None
    fenics_symbol = None

    def __init__(self, first, second):
        if not isinstance(first, Expression):
            raise ValueError(f'First operand of {type(self).__name__}({first}, {second}) must be Expression '
                             f'(given: {type(first).__name__}).')
        if not isinstance(second, Expression):
            raise ValueError(f'Second operand of {type(self).__name__}({first}, {second}) must be Expression '
                             f'(given: {type(second).__name__}).')
        if not _broadcastable_shapes(first.shape, second.shape):
            raise ValueError(f'Operands of {type(self).__name__}({first}, {second}) have incompatible shapes '
                             f'({first.shape} and {second.shape}).')

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

    def fenics_expr(self, params):
        import ufl
        if self.fenics_symbol is None:
            raise NotImplementedError(f'UFL does not support operand {self.numpy_symbol}')

        ufl_op = getattr(ufl, self.fenics_symbol) if isinstance(self.fenics_symbol, str) else self.fenics_symbol
        first = self.first.fenics_expr(params)
        second = self.second.fenics_expr(params)
        if not _broadcastable_shapes(first.shape, second.shape):
            raise ValueError(f'Incompatible shapes of expressions "{first}" and "{second}" with shapes '
                             f'{first.shape} and {second.shape} for binary operator {self.numpy_symbol}')
        return np.vectorize(ufl_op)(first, second)

    def __str__(self):
        return f'({self.first} {self.numpy_symbol} {self.second})'


class Neg(Expression):
    """Negated :class:`Expression`."""

    def __init__(self, operand):
        if not isinstance(operand, Expression):
            raise ValueError(f'Operand of {type(self).__name__}({operand}) must be Expression '
                             f'(given: {type(operand).__name__}).')
        self.operand = operand
        self.shape = operand.shape

    def numpy_expr(self):
        return f'(- {self.operand.numpy_expr()})'

    def fenics_expr(self, params):
        return np.vectorize(lambda x: -x)(self.operand.fenics_expr(params))

    def __str__(self):
        return f'(- {self.operand})'


class Indexed(Expression):
    """Indexed :class:`Expression`."""

    def __init__(self, base, index):
        if not isinstance(base, Expression):
            raise ValueError(f'Base of index expression {base}[{index}] must be Expression '
                             f'(given: {type(base).__name__}).')
        if not isinstance(index, int) and \
                not (isinstance(index, tuple) and all(isinstance(i, int) for i in index)):
            raise ValueError(f'Index of index expression {base}[{index}] must be int or tuple of ints '
                             f'(given: "{index}" of type {type(index).__name__}).')
        if isinstance(index, int):
            index = (index,)
        if not len(index) == len(base.shape):
            raise ValueError(f'Wrong number of indices for index expression {base}[{index}] '
                             f'(given {len(index)} indices for expression of shape {base.shape}).')
        for i, (ind, s) in enumerate(zip(index, base.shape)):
            if not 0 <= ind < s:
                raise ValueError(f'Invalid index for index expression {base}[{index}] '
                                 f'(given index {ind} for axis {i} of dimension {s}).')
        self.base, self.index = base, index
        self.shape = base.shape[len(index):]

    def numpy_expr(self):
        index = ['...'] + [repr(i) for i in self.index]
        return f'{self.base.numpy_expr()}[{",".join(index)}]'

    def fenics_expr(self, params):
        return np.array(self.base.fenics_expr(params)[self.index])

    def __str__(self):
        index = [str(i) for i in self.index]
        return f'{self.base}[{",".join(index)}]'


class UnaryFunctionCall(Expression):
    """Compound :class:`Expression` of an unary function applied to a sub-expression.

    The function is applied component-wise.
    """

    numpy_symbol = None
    fenics_symbol = None

    _parameters_varargs_warning = False  # silence warning due to use of *args in __init__

    def __init__(self, arg, *args):
        if args:
            raise ValueError(f'{self.numpy_symbol} takes a single argument (given: {(arg,) + args})')
        if not isinstance(arg, Expression):
            raise ValueError(f'Argument of function call {self.numpy_symbol}({arg}) must be Expression '
                             f'(given: {type(arg).__name__}).')
        self.arg = arg
        self.shape = self.arg.shape

    def numpy_expr(self):
        return f'{self.numpy_symbol}({self.arg.numpy_expr()})'

    def fenics_expr(self, params):
        import ufl
        if self.fenics_symbol is None:
            raise NotImplementedError(f'UFL does not support function {self.numpy_symbol}')
        ufl_op = getattr(ufl, self.fenics_symbol)
        return np.vectorize(ufl_op)(self.arg.fenics_expr(params))

    def _format_repr(self, max_width, verbosity):
        return super()._format_repr(max_width, verbosity, override={'args': None})

    def __str__(self):
        return f'{self.numpy_symbol}({self.arg})'


class UnaryReductionCall(UnaryFunctionCall):
    """Compound :class:`Expression` of an unary function applied to a sub-expression.

    The function is applied to the entire vector/matrix/tensor the sub-expression evaluates to,
    returning a single number.
    """

    numpy_symbol = None
    fenics_op = None

    _parameters_varargs_warning = False  # silence warning due to use of *args in __init__

    def __init__(self, arg, *args):
        super().__init__(arg, *args)
        self.shape = ()

    def numpy_expr(self):
        return (f'(lambda _a: {self.numpy_symbol}(_a.reshape(_a.shape[:-{len(self.arg.shape)}] + (-1,)), '
                f'axis=-1))({self.arg.numpy_expr()})')

    def fenics_expr(self, params):
        if self.fenics_op is None:
            raise NotImplementedError(f'UFL does not support function {self.numpy_symbol}')
        r = None
        op = self.fenics_op
        if isinstance(op, str):
            import ufl
            op = getattr(ufl, op)
        for el in self.arg.fenics_expr(params).flat:
            r = el if r is None else op(r, el)
        return np.array(r).reshape(())

    def __str__(self):
        return f'{self.numpy_symbol}({self.arg})'


def _broadcastable_shapes(first, second):
    return all(f == s or f == 1 or s == 1 for f, s in zip(first[::-1], second[::-1]))


class Sum(BinaryOp):  numpy_symbol = '+'; fenics_symbol = operator.add      # NOQA
class Diff(BinaryOp): numpy_symbol = '-'; fenics_symbol = operator.sub      # NOQA
class Prod(BinaryOp): numpy_symbol = '*'; fenics_symbol = operator.mul      # NOQA
class Div(BinaryOp):  numpy_symbol = '/'; fenics_symbol = operator.truediv  # NOQA


class Pow(BinaryOp):  numpy_symbol = '**'; fenics_symbol = 'elem_pow'    # NOQA
class LE(BinaryOp):   numpy_symbol = '<='; fenics_symbol = 'le'          # NOQA
class GE(BinaryOp):   numpy_symbol = '>='; fenics_symbol = 'ge'          # NOQA
class LT(BinaryOp):   numpy_symbol = '<';  fenics_symbol = 'lt'          # NOQA
class GT(BinaryOp):   numpy_symbol = '>';  fenics_symbol = 'gt'          # NOQA
class Mod(BinaryOp):  numpy_symbol = '%';  fenics_symbol = None          # NOQA


class sin(UnaryFunctionCall):      numpy_symbol = 'sin';     fenics_symbol = 'sin'       # NOQA
class cos(UnaryFunctionCall):      numpy_symbol = 'cos';     fenics_symbol = 'cos'       # NOQA
class tan(UnaryFunctionCall):      numpy_symbol = 'tan';     fenics_symbol = 'tan'       # NOQA
class arcsin(UnaryFunctionCall):   numpy_symbol = 'arcsin';  fenics_symbol = 'asin'      # NOQA
class arccos(UnaryFunctionCall):   numpy_symbol = 'arccos';  fenics_symbol = 'acos'      # NOQA
class arctan(UnaryFunctionCall):   numpy_symbol = 'arctan';  fenics_symbol = 'atan'      # NOQA
class sinh(UnaryFunctionCall):     numpy_symbol = 'sinh';    fenics_symbol = 'sinh'      # NOQA
class cosh(UnaryFunctionCall):     numpy_symbol = 'cosh';    fenics_symbol = 'cosh'      # NOQA
class tanh(UnaryFunctionCall):     numpy_symbol = 'tanh';    fenics_symbol = 'tanh'      # NOQA
class exp(UnaryFunctionCall):      numpy_symbol = 'exp';     fenics_symbol = 'exp'       # NOQA
class log(UnaryFunctionCall):      numpy_symbol = 'log';     fenics_symbol = 'ln'        # NOQA
class sqrt(UnaryFunctionCall):     numpy_symbol = 'sqrt';    fenics_symbol = 'sqrt'      # NOQA
class sign(UnaryFunctionCall):     numpy_symbol = 'sign';    fenics_symbol = 'sign'      # NOQA


class arcsinh(UnaryFunctionCall):
    numpy_symbol = 'arcsinh'

    def fenics_expr(self, params):
        arg = self.arg
        expr = log(arg + sqrt(arg**Constant(2) + Constant(1)))
        return expr.fenics_expr(params)


class arccosh(UnaryFunctionCall):
    numpy_symbol = 'arccosh'

    def fenics_expr(self, params):
        expr = log(self.arg + sqrt(self.arg**Constant(2) - Constant(1)))
        return expr.fenics_expr(params)


class arctanh(UnaryFunctionCall):
    numpy_symbol = 'arctanh'

    def fenics_expr(self, params):
        expr = Constant(0.5) * log((Constant(1) + self.arg) / (Constant(1) - self.arg))
        return expr.fenics_expr(params)


class exp2(UnaryFunctionCall):
    numpy_symbol = 'exp2'

    def fenics_expr(self, params):
        from ufl import elem_pow
        return np.vectorize(lambda x: elem_pow(2, x))(self.arg.fenics_expr(params))


class log2(UnaryFunctionCall):
    numpy_symbol = 'log2'

    def fenics_expr(self, params):
        expr = log(self.arg) / log(Constant(2))
        return expr.fenics_expr(params)


class log10(UnaryFunctionCall):
    numpy_symbol = 'log10'

    def fenics_expr(self, params):
        expr = log(self.arg) / log(Constant(10))
        return expr.fenics_expr(params)


class abs(UnaryFunctionCall):
    numpy_symbol = 'abs'

    def fenics_expr(self, params):
        from ufl.algebra import Abs
        return np.vectorize(lambda x: Abs(x))(self.arg.fenics_expr(params))


class angle(UnaryFunctionCall):
    numpy_symbol = 'angle'
    fenics_op = None

    def __init__(self, arg):
        if arg.shape[-1] != 2:
            raise ValueError
        self.arg = arg
        self.shape = arg.shape[:-1]

    def fenics_expr(self, params):
        if len(self.shape) > 1:
            raise NotImplementedError
        import ufl
        import dolfin
        arg = self.arg.fenics_expr(params)
        assert arg.shape == (2,)
        return np.array(
            [ufl.atan_2(arg[1], arg[0]) + ufl.lt(ufl.atan_2(arg[1], arg[0]), 0) * dolfin.Constant(2*np.pi)]
        ).reshape(())


class min(UnaryReductionCall):  numpy_symbol = 'min';  fenics_op = 'min_value'   # NOQA
class max(UnaryReductionCall):  numpy_symbol = 'max';  fenics_op = 'max_value'   # NOQA
class sum(UnaryReductionCall):  numpy_symbol = 'sum';  fenics_op = operator.add  # NOQA
class prod(UnaryReductionCall): numpy_symbol = 'prod'; fenics_op = operator.mul  # NOQA


class norm(UnaryReductionCall):
    numpy_symbol = 'norm'

    def fenics_expr(self, params):
        return sqrt(sum(self.arg**Constant(2))).fenics_expr(params)


class Pi(BaseConstant): numpy_symbol = 'pi'; fenics_symbol = 'pi'  # NOQA
class E(BaseConstant):  numpy_symbol = 'e';  fenics_symbol = 'e'   # NOQA


pi = Pi()
e  = E()


_numpy_functions = {k: getattr(np, k) for k in {'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2',
                                                'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
                                                'exp', 'exp2', 'log', 'log2', 'log10', 'sqrt', 'abs', 'sign',
                                                'min', 'max', 'sum', 'prod',
                                                'pi', 'e',
                                                'array', 'broadcast_arrays', 'moveaxis', 'newaxis'}}

_numpy_functions['norm']  = np.linalg.norm
_numpy_functions['angle'] = lambda x: np.arctan2(x[..., 1], x[..., 0]) % (2*np.pi)  # np.angle uses different convention
