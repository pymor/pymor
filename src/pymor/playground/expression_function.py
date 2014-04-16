# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core.interfaces import BasicInterface

import numpy as np

try:
    import sympy
    HAVE_SYMPY = True
except ImportError:
    HAVE_SYMPY = False


class ExpressionFunction(BasicInterface):

    def __init__(self, expressions, variables='x y z'):
        if not HAVE_SYMPY:
            raise ImportError('could not import sympy')
        variables = variables.split(' ')
        self._variables = sympy.symbols(variables)
        self._var_string = variables
        self._expressions = [sympy.sympify(e) for e in expressions]

    def __call__(self, domain_vec):
        assert len(self._variables) == len(domain_vec)
        subs = {self._variables[i]: domain_vec[i] for i in range(len(domain_vec))}
        return np.array([e.evalf(subs=subs) for e in self._expressions])


if __name__ == "__main__":
    A = ExpressionFunction(['x**2'], 'x')
    B = ExpressionFunction(['sin(y)', 'x'], 'x y')
    import pylab
    for f in range(-2, 4):
        pylab.plot(f, A([f]), 'x')
        pylab.plot(f, B([f, f])[0], '|')
        pylab.plot(f, B([f, f])[1], '-')
    pylab.show()
