# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number
from itertools import zip_longest
import numpy as np

from pymor.parameters.base import ParametricObject


builtin_max = max


def parse_expression(expression, dim_domain, parameters={}, values={}):
    if isinstance(expression, Expression):
        return expression
    locals_dict = {name: Parameter(name, dim) for name, dim in parameters.items()}
    locals_dict['x'] = X(dim_domain)
    return _convert_to_expression(eval(expression, dict(globals(), **values), locals_dict))


class Expression(ParametricObject):

    shape = None

    def to_numpy(self):
        expression = self.numpy_expr()
        code = compile(expression, '<expression>', 'eval')
        return (
            lambda x, mu={}: (np.broadcast_to(eval(code,
                                                   dict(input_shape=x.shape[:-1], **_numpy_functions),
                                                   dict(mu, x=x, mu=mu, input_shape=x.shape[:-1])),
                                              x.shape[:-1] + self.shape))
        )

    def numpy_expr(self):
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

    numpy_symbol = None
    shape = ()

    def numpy_expr(self):
        return f'array({self.numpy_symbol}, ndmin=len(input_shape)+{len(self.shape)})'

    def __str__(self):
        return str(self.numpy_symbol)


class Parameter(BaseConstant):

    def __init__(self, name, dim):
        self.name, self.dim = name, dim
        self.numpy_symbol = name
        self.shape = (dim,)
        self.parameters_own = {name: dim}


class Constant(BaseConstant):

    def __init__(self, value):
        self.value = value = np.array(value)
        self.numpy_symbol = repr(value.tolist())
        self.shape = value.shape


class BinaryOp(Expression):

    numpy_symbol = None

    def __init__(self, first, second):
        if not _broadcastable(first.shape, second.shape):
            raise ValueError('Incompatible shapes')
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

    def __init__(self, operand):
        self.operand = operand
        self.shape = operand.shape

    def numpy_expr(self):
        return f'(- {self.operand.numpy_expr()})'

    def __str__(self):
        return f'(- {self.operand})'


class Indexed(Expression):

    def __init__(self, base, index):
        self.base, self.index = base, index
        self.shape = base.shape[(1 if isinstance(index, Number) else len(index)):]

    def numpy_expr(self):
        index = [repr(self.index)] if isinstance(self.index, Number) else [repr(i) for i in self.index]
        index = ['...'] + index
        return f'{self.base.numpy_expr()}[{",".join(index)}]'

    def __str__(self):
        return f'{self.base}[{",".join(self.index)}]'


class X(Expression):

    def __init__(self, dim_domain):
        self.dim_domain = dim_domain
        self.shape = (dim_domain,)

    def numpy_expr(self):
        return 'x'

    def __str__(self):
        return 'x'


class UnaryFunctionCall(Expression):

    numpy_symbol = None

    def __init__(self, arg):
        self.arg = arg
        self.shape = arg.shape

    def numpy_expr(self):
        return f'{self.numpy_symbol}({self.arg.numpy_expr()})'

    def __str__(self):
        return f'{self.numpy_symbol}({self.arg})'


class UnaryReductionCall(Expression):

    def __init__(self, arg):
        self.arg = arg
        self.shape = ()

    def numpy_expr(self):
        return f'{self.numpy_symbol}({self.arg.numpy_expr()}.reshape(input_shape + (-1,)), axis=-1)'

    def __str__(self):
        return f'{self.numpy_symbol}({self.arg})'


def _convert_to_expression(obj):
    if isinstance(obj, Expression):
        return obj
    elif isinstance(obj, Number):
        return Constant(obj)
    else:
        obj = np.array(obj)
        if obj.dtype == object:
            raise NotImplementedError
        return Constant(obj)


def _broadcastable(first, second):
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
class low(UnaryFunctionCall):      numpy_symbol = 'log'      # NOQA
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
                                                'array', 'newaxis'}}

_numpy_functions['norm']  = np.linalg.norm
_numpy_functions['angle'] = lambda x: np.arctan2(x[..., 1], x[..., 0]) % (2*np.pi)
